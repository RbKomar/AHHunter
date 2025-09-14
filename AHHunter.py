import re
import threading
import time
import os
import platform
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

API_KEY = os.getenv('HYPIXEL_API_KEY')
if not API_KEY:
    print("Warning: HYPIXEL_API_KEY environment variable not set.")

thread_lock = threading.Lock()
AUCTION_DATA = {}
last_updated_old = 0


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 200)
    else:
        print("\007", end="")


def get_number_of_auctions_pages_and_if_updated():
    api_auctions_url = 'https://api.hypixel.net/skyblock/auctions'
    try:
        response = requests.get(api_auctions_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        number_of_pages = data["totalPages"]
        if number_of_pages > 120:
            raise Exception("Abusing Hypixel API")
        last_updated = data["lastUpdated"]
        return number_of_pages, last_updated
    except requests.exceptions.RequestException as e:
        print(f"Error fetching number of pages: {e}")
        return 0, 0


def get_last_updated():
    api_auctions_url = 'https://api.hypixel.net/skyblock/auctions'
    response = requests.get(api_auctions_url)
    response = response.json()
    last_updated = response["lastUpdated"]
    return last_updated


def get_auctions(item_name: str, price: int, page: int, reforges_list, lore='', flipper_mode=True):
    api_auctions_url = 'https://api.hypixel.net/skyblock/auctions'
    parameters = {"page": page}
    try:
        response = requests.get(api_auctions_url, params=parameters, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        # print(f"Skipping page {page} due to request error: {e}")
        return

    for auction in data.get("auctions", []):
        if not flipper_mode:
            if item_name in auction["item_name"].lower() and not auction["claimed"] \
                    and lore in auction["item_lore"]:
                try:
                    if auction["bin"]:
                        if auction["starting_bid"] < price:
                            print(item_name, " price: " + str(auction['starting_bid']))
                            beep()
                except KeyError:
                    continue
        else:
            try:
                if auction["bin"]:
                    item_name_local = auction["item_name"].lower()
                    item_name_local = re.sub(r'(\[\w*\s\d*\])', '', item_name_local)  # [lvl xx]
                    item_name_local = re.sub(r'\s\s+', ' ', item_name_local)  # double spaces to one
                    item_name_local = re.sub(r'([^\w\s]\W*$)', '', item_name_local, re.MULTILINE)  # *** at the end of the name
                    item_name_local = re.sub(r'(^\W\s)', '', item_name_local,
                                       re.MULTILINE)  # this weird umbrella ect at the beginning
                    reforges_regex = re.compile(r'\b(?:' + r'|'.join(map(re.escape, reforges_list)) + r")\b", re.IGNORECASE | re.MULTILINE)
                    item_name_local = re.sub(reforges_regex, '', item_name_local)  # deleting reforges
                    item_name_local = item_name_local.strip()
                    if item_name_local == "enchanted book":
                        lore_local = auction['item_lore']
                        book_names = lore_local.split('\n')[0].split(',')
                        legendary_enchantment = False
                        for names in book_names:
                            enchantments = names.split('9')
                            for enchantment in enchantments:
                                if "§l" in enchantment:
                                    item_name_local = enchantment.strip('§d§l§7§l,').split('\n')[0]
                                    legendary_enchantment = True
                            if not legendary_enchantment:
                                if len(enchantments) > 1:
                                    item_name_local = enchantments[1].strip('§9§d§l§7§l,').split('\n')[0]
                                else:
                                    item_name_local = enchantments[0].strip('§9§d§l§7§l,').split('\n')[0]
                        if "Use this on" in item_name_local or len(item_name_local) < 2:
                            continue
                        item_name_local = item_name_local.strip(" \n")

                    with thread_lock:
                        global AUCTION_DATA
                        if item_name_local.lower() not in AUCTION_DATA.keys():
                            AUCTION_DATA[item_name_local.lower()] = [
                                str(auction['starting_bid']) + '|' + str(auction['uuid'])]
                        else:
                            AUCTION_DATA[item_name_local.lower()].append(
                                str(auction['starting_bid']) + '|' + str(auction['uuid']))
            except KeyError:
                continue


def rename_to_similar(item_name, matches_df):
    similar_words = pd.DataFrame()
    similar_words["index_left"] = matches_df["left_side"].isin([item_name])
    similar_words["index_right"] = matches_df["right_side"].isin([item_name])
    if similar_words["index_left"].sum() != 0:
        return matches_df[similar_words["index_left"][0]]
    elif similar_words["index_right"].sum() != 0:
        return matches_df[similar_words["index_right"][0]]
    else:
        return item_name


def auction_finder(reforges_list, item_name: str = '', price: int = 0, lore=''):
    number_of_pages, last_updated = get_number_of_auctions_pages_and_if_updated()
    global last_updated_old
    if last_updated == last_updated_old or number_of_pages == 0:
        return

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_auctions, item_name, price, page, reforges_list, lore) for page in range(number_of_pages)]
        for future in futures:
            future.result()  # Wait for all threads to complete

    last_updated_old = last_updated


def MAD_Z_Score(data, consistency_correction=1.4826):
    if np.sum(data) == 0:
        return np.array([]), 0
    median = np.median(data)
    mean = np.mean(data)
    deviation_from_med = np.abs(np.array(data) - median)
    deviation_from_mean = np.abs(np.array(data) - mean)
    MeanAD = np.mean(deviation_from_mean)
    MAD = np.median(deviation_from_med)
    if MAD != 0:
        MAD_zscore = deviation_from_med / (consistency_correction * MAD)
        return MAD_zscore, MAD
    else:
        MeanAD_zscore = deviation_from_mean / (np.mean(deviation_from_mean) * 1.253314)
        return MeanAD_zscore, MeanAD


def find_items_to_flip(data: pd.DataFrame()):
    flip_items = {}
    if data.empty:
        return

    for index, row in data.iterrows():
        # Correctly handle the Series by getting its name (the item name) and values
        item_name_str = index
        # Drop NaN values which represent missing auction data for this item
        product_series = row.dropna()
        if product_series.empty:
            continue

        # Separate prices and UUIDs
        try:
            prices = product_series.str.split('|').str[0].astype(float)
            uuids = product_series.str.split('|').str[1]
        except (AttributeError, IndexError):
            # print(f"Warning: Skipping row for item '{item_name_str}' due to malformed data.")
            continue
        
        product = prices.to_numpy()

        if len(product) <= 1 or product.max() - product.min() == 0:
            continue
        
        product_normalize = (product - product.min()) / (product.max() - product.min())
        
        mad_zscore, mad = MAD_Z_Score(product_normalize)
        product_median = np.median(product)
        product_anomalies = (mad_zscore > 3)

        if 0 < np.sum(product_anomalies) < 3:
            product_sorted = sorted(product)
            for idx, anom in enumerate(product_anomalies):
                if anom and product[idx] < product_median:
                    outlier = product[idx]
                    product_sorted.remove(outlier)
                    if not product_sorted:
                        continue
                    cheapest = product_sorted[0]
                    expected_profit = cheapest - outlier
                    flip_items[item_name_str] = [outlier, cheapest, expected_profit, len(product), uuids.iloc[idx]]

    if not flip_items:
        return

    items_to_flip_dataset = pd.DataFrame.from_dict(flip_items, orient="index",
                                                   columns=["Hunted Price", "LBin", "Expected Profit",
                                                            "Items on market", "Auction uuid"])
    if not items_to_flip_dataset.empty:
        items_to_flip_dataset.sort_values(by="Expected Profit", ascending=False, inplace=True)
        print(items_to_flip_dataset)
    
    global AUCTION_DATA
    AUCTION_DATA = {}


def check_auctions(reforges_list, job_id: int):
    print(f"Thread #{job_id} starting job...")
    auction_finder(reforges_list)
    data = pd.DataFrame.from_dict(AUCTION_DATA, orient='index')
    find_items_to_flip(data)
    print(f"Thread #{job_id} has finished its job.")


if __name__ == '__main__':
    reforges_list = []
    try:
        reforges_df = pd.read_csv("reforges.csv")
        reforges_list = reforges_df["reforges"].str.lower().to_list()
    except FileNotFoundError:
        print("Warning: reforges.csv not found. Reforge removal will be skipped.")
    except Exception as e:
        print(f"Error loading reforges.csv: {e}")

    thread_counter = 0
    start_time = time.time()
    
    while True:
        thread_counter += 1
        check_auctions(reforges_list, thread_counter)
        
        if (time.time() - start_time) > 145 * 10:
            print("Completed a cycle of 10 runs. Exiting or pausing as placeholder.")
            time.sleep(60)
            start_time = time.time() 

        time.sleep(35)
