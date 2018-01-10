'''
https://github.com/ace19-dev/Categorize-ecommerce-photos
make_bson.py --> split_collection_per_class.py --> make_dataset_dir.py --> train.py

내꺼
1. https://github.com/aimmvp/kaggleMaster/blob/master/Cdiscount/mongo_script.md
  - train.bson import

2. ace19_split_collection.py
  - Make index for train.bson
  - split collection per category_id only 300 collections

3. ace19_make_bson.py
  - make bson file at file system per category_id
'''


import pymongo
from pymongo import MongoClient
import numpy as np
from bson import BSON

np.random.seed(777)  # for reproducibility

client = MongoClient()
db = client.train

collections = db.collection_names()

print("collection count : ", len(collections))   # 102
for item in collections:
    if item.find('train_') is -1: # Oraginal Collection(trian) is skiped
        continue

    col = db[item]
    print("item : ", item)
    # col.drop()
    file_name = "../../ace19/input/" + item + ".bson"
    with open(file_name, 'wb+') as f:
        for doc in col.find():
            f.write(BSON.encode(doc))