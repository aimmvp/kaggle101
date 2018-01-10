'''
https://github.com/ace19-dev/Categorize-ecommerce-photos
make_bson.py --> split_collection_per_class.py --> make_dataset_dir.py --> train.py

내꺼
1. https://github.com/aimmvp/kaggleMaster/blob/master/Cdiscount/mongo_script.md
  - train.bson import

2. ace19_split_collection.py
  - Make index for train.bson
  - split collection per category_id only 300 collections
'''


import pymongo
from pymongo import MongoClient
import numpy as np
import pandas as pd
import random

np.random.seed(777)  # for reproducibility

client = MongoClient()
db = client.train

train_collection = db.train


# create a index by category_id
# train_collection.create_index("category_id")
# print(" Count of train_collection : ", train_collection.count())
# for index in train_collection.list_indexes():
#     print(index)

csv_file = "input/category_names.csv"
csv_data = pd.read_csv(csv_file, usecols=[0], skiprows=range(0,10))
category_ids = csv_data.values

i_cnt = 0
cate_cnt = 0
for category_id in category_ids:
    i_collection_id = int(category_id[0])
    s_collection_id = str(category_id[0])

    item_cnt = train_collection.find({'category_id': i_collection_id}).count()
    if item_cnt > 200 and item_cnt < 300 and cate_cnt < 21:
        collection = db['train_' + s_collection_id]
        # collection.drop()
        items = train_collection.find({'category_id': i_collection_id}).limit(100)
        collection.insert_many(items)
        cate_cnt += 1
        print("[", cate_cnt, "] train_", s_collection_id, " : ", item_cnt)
