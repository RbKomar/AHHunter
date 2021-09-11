import re
import threading
import time
import winsound

import numpy as np
import pandas as pd
import requests

API_KEY = 'xxx'
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


def get_number_of_auctions_pages_and_if_updated():
    api_auctions_url = 'https://api.hypixel.net/skyblock/auctions'
    response = requests.get(api_auctions_url)
    response = response.json()
    number_of_pages = response["totalPages"]
    if number_of_pages > 120:
        raise Exception("Abusing hypixel Api")
    last_updated = response["lastUpdated"]
    return number_of_pages, last_updated


def get_last_updated():
    api_auctions_url = 'https://api.hypixel.net/skyblock/auctions'
    response = requests.get(api_auctions_url)
    response = response.json()
    last_updated = response["lastUpdated"]
    return last_updated


def get_auctions(item_name: str, price: int, page: int, reforges_list, matches, lore='', flipper_mode=True):
    api_auctions_url = 'https://api.hypixel.net/skyblock/auctions'
    parameters = {"page": page}
    response = requests.get(api_auctions_url, params=parameters).json()
    for auction in response["auctions"]:
        if not flipper_mode:
            if item_name in auction["item_name"].lower() and not auction["claimed"] \
                    and lore in auction["item_lore"]:
                try:
                    if auction["bin"]:
                        if auction["starting_bid"] < price:
                            print(print(item_name), " price: " + str(auction['starting_bid']))
                            for i in range(1, 10):
                                winsound.Beep(i * 100, 200)
                except KeyError as k:
                    continue
        else:
            try:
                if auction["bin"]:
                    item_name = auction["item_name"].lower()
                    item_name = re.sub(r'(\[\w*\s\d*\])', '', item_name)  # [lvl xx]
                    item_name = re.sub(r'\s\s+', ' ', item_name)  # double spaces to one
                    item_name = re.sub(r'([^\w\s]\W*$)', '', item_name, re.MULTILINE)  # *** at the end of the name
                    item_name = re.sub(r'(^\W\s)', '', item_name,
                                       re.MULTILINE)  # this weird umbrella ect at the beginning
                    reforges_regex = re.compile(r'\b(?:' + r'|'.join(reforges_list) + r")\b", re.MULTILINE)
                    item_name = re.sub(reforges_regex, '', item_name)  # deleting reforges
                    item_name = item_name.strip()
                    if item_name == "enchanted book":
                        lore = auction['item_lore']
                        book_names = lore.split('\n')[0].split(',')
                        legendary_enchantment = False
                        for names in book_names:
                            enchantments = names.split('9')
                            for enchantment in enchantments:
                                if "§l" in enchantment:
                                    item_name = enchantment.strip('§d§l§7§l,').split('\n')[0]
                                    legendary_enchantment = True
                            if not legendary_enchantment:
                                if len(enchantments) > 1:
                                    item_name = enchantments[1].strip('§9§d§l§7§l,').split('\n')[0]
                                else:
                                    item_name = enchantments[0].strip('§9§d§l§7§l,').split('\n')[0]
                        if "Use this on" in item_name or len(item_name) < 2:
                            continue
                        item_name = item_name.strip(" \n")
                    # item_name = rename_to_similar(item_name, matches) #too slow - maybe i'll optimize in the future
                    with thread_lock:
                        global AUCTION_DATA
                        if item_name.lower() not in AUCTION_DATA.keys():
                            AUCTION_DATA[item_name.lower()] = [
                                str(auction['starting_bid']) + '|' + str(auction['uuid'])]
                        else:
                            AUCTION_DATA[item_name.lower()].append(
                                str(auction['starting_bid']) + '|' + str(auction['uuid']))
            except KeyError as k:
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


def auction_finder(reforges_list, matches, item_name: str = '', price: int = 0, lore=''):
    threads = []
    number_of_pages, last_updated = get_number_of_auctions_pages_and_if_updated()
    global last_updated_old
    if last_updated == last_updated_old:
        last_updated_old = last_updated_old
    for page in range(0, number_of_pages):
        x = threading.Thread(target=get_auctions, args=(item_name, price, page, reforges_list, matches, lore))
        threads.append(x)
        x.start()
    for thread in threads:
        thread.join()
    last_updated_old = last_updated
    return last_updated


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
    for index, row in data.iterrows():
        product = row.drop(columns=row[0]).to_numpy()
        products_uuid = None
        with np.nditer([product, products_uuid], ['refs_ok'], [['readwrite'], ['writeonly', 'allocate']]) as it:
            while not it.finished:
                if it[0] == None:
                    it.iternext()
                    continue
                it[1] = str(it[0]).split('|')[1]
                it[0] = float(str(it[0]).split('|')[0])
                it.iternext()
            product = it.operands[0]
            products_uuid = it.operands[1]
        product = product[~np.isnan(product.astype(float))].astype(float)
        if product.max() - product.min() == 0:
            continue
        product_normalize = (product - product.min()) / (product.max() - product.min())
        if len(product) > 1:
            mad_zscore, mad = MAD_Z_Score(product_normalize)
            product_median = np.median(product)
            product_anomalies = (mad_zscore > 3)
            if 0 < np.sum(product_anomalies) < 3:
                product_sorted = sorted(product)
                for idx, anom in enumerate(product_anomalies):
                    if anom and product[idx] < product_median:
                        outlier = product[idx]
                        product_sorted.remove(product[idx])
                        cheapest = product_sorted[0]
                        expected_profit = cheapest - outlier
                        flip_items[str(index)] = [outlier, cheapest, expected_profit, len(product), products_uuid[idx]]
    items_to_flip_dataset = pd.DataFrame.from_dict(flip_items, orient="index",
                                                   columns=["Price", "Cheapest", "Expected Profit",
                                                            "Demand", "Auction uuid"])
    items_to_flip_dataset.sort_values(by="Expected Profit", ascending=False, inplace=True)
    print(items_to_flip_dataset)
    global AUCTION_DATA
    AUCTION_DATA = {}


def check_auctions(reforges_list, matches: list, id: int):
    start = time.time()
    auction_finder(reforges_list, matches)
    end = time.time()
    print(end - start)
    data = pd.DataFrame.from_dict(AUCTION_DATA, orient='index')
    find_items_to_flip(data)
    print("Thread #" + str(id) + " has finished its job")


if __name__ == '__main__':
    last_updated = 0
    path = "auction_objects_data.csv"
    reforges_list = ((pd.read_csv("reforges.csv"))["reforges"]).to_list()
    reforges_list_lower = []
    for reforge in reforges_list:
        reforges_list_lower.append(reforge.lower())
    reforges_list = reforges_list + reforges_list_lower
    start = time.time()
    agents = []
    thread_counter = 0
    while True:
        print("Creating new thread")
        thread_counter += 1
        s1 = threading.Thread(target=check_auctions, args=[reforges_list, [], thread_counter])
        s1.start()
        agents.append(s1)
        agents[-1].join()
        time.sleep(35)
        end = time.time()
        if end - start % 145 == 0:
            thread_to_kill = agents.pop(0)
