import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from bson import BSON
import csv
'''
https://velopert.com/560
db.train.createIndex({category_id:1})
'''

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

'''
count_collection = train_db.count_collection
'''
# 2. Get category names from category_names.csv
category_names = pd.read_csv('input/category_names.csv', usecols=[0], dtype=int)
category_ids = category_names.values

# 3. Make bson files by category_id(train_#######.bson) limit 10
MAX_ITEM_CNT = 100
count = 0
under_dic = {}
for cate_id in category_ids:
    count = count + 1
    cate_id = str(cate_id)[1:11]
    file_name = './input/train_' + cate_id+ '.bson'

    with open(file_name, 'wb+') as f:
        for item in train_collection.find({'category_id': int(cate_id)}).limit(MAX_ITEM_CNT):
            item_cnt = train_collection.find({'category_id': int(cate_id)}).count()
            print("[", count, "] id : ", file_name, "item_cnt : ", item_cnt)
            if item_cnt < MAX_ITEM_CNT:
                under_dic[str(cate_id)] = str(item_cnt)
            f.write(BSON.encode(item))


print("############### TOTAL CNT : ", count)

# 4. Make category_id and count list Under limit count category_id list
with open("./input/under_list.csv", "w") as f:
    w = csv.writer(f)
    w.writerows(under_dic.items())
# 1780개

# 5. base on under_list.csv save image file

# 6. Augumentation and make bson file

# 7. merge bson file to train_merge.bson

# 8. Train model using train_merge.bson