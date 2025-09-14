import logging
import os
import platform
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from database import DatabaseManager

CONFIG = {
    "api_key": os.getenv('HYPIXEL_API_KEY'),
    "max_api_pages": 120,
    "request_timeout": 10,
    "max_workers": 10,
    "scan_interval_seconds": 35,
    "z_score_threshold": 3.0,
    "min_outlier_count": 1,
    "max_outlier_count": 3,
    "reforges_csv_path": "reforges.csv",
    "results_csv_path": "flips.csv",
    "db_path": "ah_prices.db",
    "log_level": "INFO",
}

logging.basicConfig(
    level=CONFIG["log_level"],
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("ahhunter.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class AuctionHunter:
    def __init__(self, config):
        self.config = config
        self.thread_lock = threading.Lock()
        self.auction_data = {}
        self.last_updated = 0
        self.reforges = self._load_reforges()
        self.db_manager = DatabaseManager(db_path=self.config["db_path"])

        if not self.config["api_key"]:
            logging.warning("HYPIXEL_API_KEY environment variable not set. API requests may fail.")

    def _load_reforges(self):
        try:
            reforges_df = pd.read_csv(self.config["reforges_csv_path"])
            return reforges_df["reforges"].str.lower().to_list()
        except FileNotFoundError:
            logging.warning(f"{self.config['reforges_csv_path']} not found. Reforge removal will be skipped.")
        except Exception as e:
            logging.error(f"Error loading {self.config['reforges_csv_path']}: {e}")
        return []

    @staticmethod
    def _beep():
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 200)
        else:
            print("\007", end="", flush=True)

    def _get_total_pages(self):
        api_url = 'https://api.hypixel.net/skyblock/auctions'
        try:
            response = requests.get(api_url, timeout=self.config["request_timeout"])
            response.raise_for_status()
            data = response.json()
            total_pages = data.get("totalPages", 0)
            
            if total_pages > self.config["max_api_pages"]:
                raise Exception(f"API returned {total_pages} pages, which is above the configured max of {self.config['max_api_pages']}.")

            self.last_updated = data.get("lastUpdated", 0)
            return total_pages, self.last_updated
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching total auction pages: {e}")
            return 0, 0

    def _fetch_auction_page(self, page):
        api_url = 'https://api.hypixel.net/skyblock/auctions'
        params = {"page": page}
        try:
            response = requests.get(api_url, params=params, timeout=self.config["request_timeout"])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            logging.debug(f"Skipping page {page} due to request error: {e}")
            return None

    def _process_auctions(self, data):
        for auction in data.get("auctions", []):
            try:
                if not auction.get("bin"):
                    continue

                item_name = auction.get("item_name", "").lower()
                item_name = re.sub(r'(\[\w*\s\d*\])', '', item_name)
                item_name = re.sub(r'\s{2,}', ' ', item_name)
                item_name = re.sub(r'[^\w\s]$', '', item_name, flags=re.MULTILINE)
                item_name = re.sub(r'^\W\s', '', item_name, flags=re.MULTILINE)
                
                if self.reforges:
                    reforges_regex = re.compile(r'\b(?:' + r'|'.join(map(re.escape, self.reforges)) + r")\b", re.IGNORECASE)
                    item_name = reforges_regex.sub('', item_name).strip()

                if item_name == "enchanted book":
                    lore = auction.get('item_lore', '')
                    first_line = lore.split('\n')[0]
                    item_name = first_line.split('§9')[-1].replace('§d', '').replace('§l', '').replace('§7', '').strip()

                if "use this on" in item_name.lower() or len(item_name) < 2:
                    continue
                item_name = item_name.strip()

                with self.thread_lock:
                    price_uuid = f"{auction['starting_bid']}|{auction['uuid']}"
                    if item_name not in self.auction_data:
                        self.auction_data[item_name] = [price_uuid]
                    else:
                        self.auction_data[item_name].append(price_uuid)
            except (KeyError, IndexError) as e:
                logging.warning(f"Skipping an auction due to malformed data: {e}")
                continue

    def run_auction_scan(self):
        last_updated_snapshot = self.last_updated
        total_pages, new_last_updated = self._get_total_pages()
        
        if new_last_updated == last_updated_snapshot or total_pages == 0:
            logging.info("No new auction data. Skipping scan.")
            return

        self.auction_data.clear()

        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            futures = [executor.submit(self._fetch_auction_page, page) for page in range(total_pages)]
            for future in futures:
                page_data = future.result()
                if page_data:
                    self._process_auctions(page_data)

    @staticmethod
    def _mad_z_score(data, consistency_correction=1.4826):
        if not data.any():
            return np.array([]), 0
            
        median = np.median(data)
        deviation_from_med = np.abs(data - median)
        mad = np.median(deviation_from_med)
        
        if mad != 0:
            return deviation_from_med / (consistency_correction * mad), mad
        
        # Fallback to MeanAD if MAD is 0
        mean = np.mean(data)
        deviation_from_mean = np.abs(data - mean)
        mean_ad = np.mean(deviation_from_mean)
        return deviation_from_mean / (mean_ad * 1.253314), mean_ad

    def find_and_report_flips(self):
        if not self.auction_data:
            logging.info("No auction data to analyze.")
            return
            
        data = pd.DataFrame.from_dict(self.auction_data, orient='index')
        flip_items = {}

        for item_name, row in data.iterrows():
            product_series = row.dropna()
            if product_series.empty:
                continue

            try:
                prices = product_series.str.split('|').str[0].astype(float)
                uuids = product_series.str.split('|').str[1]
            except (AttributeError, IndexError):
                logging.warning(f"Skipping row for item '{item_name}' due to malformed data.")
                continue
            
            product = prices.to_numpy()

            if len(product) <= 1 or product.max() == product.min():
                continue
            
            normalized_product = (product - product.min()) / (product.max() - product.min())
            
            z_scores, _ = self._mad_z_score(normalized_product)
            median_price = np.median(product)
            is_anomaly = z_scores > self.config["z_score_threshold"]
            
            num_anomalies = np.sum(is_anomaly)
            if self.config["min_outlier_count"] <= num_anomalies < self.config["max_outlier_count"]:
                product_sorted = sorted(product)
                for idx, is_anom in enumerate(is_anomaly):
                    if is_anom and product[idx] < median_price:
                        outlier = product[idx]
                        product_sorted.remove(outlier)
                        if not product_sorted: continue
                        
                        cheapest_non_outlier = product_sorted[0]
                        profit = cheapest_non_outlier - outlier
                        flip_items[item_name] = [outlier, cheapest_non_outlier, profit, len(product), uuids.iloc[idx]]
        
        if not flip_items:
            logging.info("No profitable flips found in this scan.")
            return

        columns = ["Hunted Price", "LBin", "Expected Profit", "Items on market", "Auction uuid"]
        results_df = pd.DataFrame.from_dict(flip_items, orient="index", columns=columns)
        results_df.sort_values(by="Expected Profit", ascending=False, inplace=True)
        
        logging.info(f"Found {len(results_df)} potential flips:\n{results_df}")
        self._save_results(results_df)

    def _save_results(self, results_df):
        try:
            file_exists = os.path.isfile(self.config["results_csv_path"])
            results_df.to_csv(self.config["results_csv_path"], mode='a', header=not file_exists, index_label="Item Name")
            logging.info(f"Saved results to {self.config['results_csv_path']}")
        except IOError as e:
            logging.error(f"Could not save results to CSV: {e}")

    def _update_price_history(self):
        """
        Iterates through the latest auction scan data and updates the database
        with the lowest BIN price for each item if it has changed.
        """
        if not self.auction_data:
            logging.info("No auction data to update price history.")
            return

        logging.info("Updating price history in the database...")
        updated_count = 0
        for item_name, auctions in self.auction_data.items():
            if not auctions:
                continue
            
            try:
                # Calculate the lowest BIN (LBIN) as the representative price
                prices = [float(a.split('|')[0]) for a in auctions]
                lbin = min(prices)
                
                if self.db_manager.update_price_if_changed(item_name, lbin):
                    updated_count += 1
            except (ValueError, IndexError) as e:
                logging.warning(f"Could not parse price for {item_name}: {e}")
                continue
        logging.info(f"Price history updated for {updated_count} items.")
            
    def run(self):
        job_id = 0
        try:
            while True:
                job_id += 1
                logging.info(f"Starting scan job #{job_id}...")
                self.run_auction_scan()
                self._update_price_history()
                self.find_and_report_flips()
                logging.info(f"Scan job #{job_id} finished. Waiting {self.config['scan_interval_seconds']} seconds.")
                time.sleep(self.config["scan_interval_seconds"])
        except KeyboardInterrupt:
            logging.info("Shutdown signal received.")
        finally:
            self.db_manager.close()


if __name__ == '__main__':
    hunter = AuctionHunter(CONFIG)
    hunter.run()
