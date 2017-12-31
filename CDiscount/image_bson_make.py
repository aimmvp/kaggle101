import pymongo
from pymongo import MongoClient
import pandas as pd
import csv
from bson import BSON
import os

# 1. Define DB, Collection
# original DB : train
# original Collection : train --> train_collection


# create a index by category_id
# train_collection.create_index("category_id")
# print(" Count of train_collection : ", train_collection.count())
# for index in train_collection.list_indexes():
#     print(index)

client = MongoClient()
train_db = client.train
train_collection = train_db.train


# 2. Get category names from category_names.csv
category_names = pd.read_csv('input/category_names.csv', usecols=[0], dtype=int)
category_ids = category_names.values
item_cnt = 0
MIN_CNT, MAX_CNT = 500, 900
split_cnt = 0
dataset_file_path_base = '../../CDiscount/input/'
dataset_file_name = ""

if not os.path.exists(dataset_file_path_base):
    os.makedirs(dataset_file_path_base)

for cate_id in category_ids:
    c_id = cate_id[0]
    item_cnt = train_collection.find({'category_id': int(cate_id)}).count()
    if int(item_cnt) > MIN_CNT and (item_cnt) < MAX_CNT:
        split_cnt += 1
        dataset_file_name = dataset_file_path_base + "train_" + str(c_id) + ".bson"
        print("MAKE_DATASET_BSON : ", dataset_file_name, " / item_cnt : ", item_cnt)
        with open(dataset_file_name, 'wb+') as f:
            for doc in train_collection.find({'category_id': int(c_id)}):
                f.write(BSON.encode(doc))

print(MIN_CNT, " ~ ", MAX_CNT, " CNT : ", split_cnt)
# 500  ~  5000  CNT :  1361
# 500  ~  900  CNT :  501



