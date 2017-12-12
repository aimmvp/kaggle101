import pymongo
import pandas as pd
import os, io
import numpy as np
from pymongo import MongoClient
from scipy.misc import imread
from keras.preprocessing.image import ImageDataGenerator

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
MAX_ITEM_CNT = 100
category_names = pd.read_csv('input/category_names.csv', usecols=[0], dtype=int)
category_ids = category_names.values

data_aug_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=10,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.7,
                                  zoom_range=[0.9, 2.2],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')


def image_augumentation(images_ndarray, category_dir):
    batch_idx = 1     # batch index

    for batch in data_aug_gen.flow(images_ndarray,
                                   batch_size=1,
                                   save_to_dir=category_dir,
                                   save_prefix='',
                                   save_format='jpeg'):
        batch_idx += 1
        # if idx > MAX_ITEM_CNT - len(y_train):
        # print("################## ", batch_idx, "/", MAX_ITEM_CNT, "/", images_ndarray.shape[0])
        # if batch_idx > MAX_ITEM_CNT-images_ndarray.shape[0]:
        if batch_idx > MAX_ITEM_CNT:
            break

# 3. Make bson files by category_id(train_#######.bson) limit MAX_ITEM_CNT
category_ids_cnt = 0
# under_dic = {}
images = []
for cate_id in category_ids:
    category_ids_cnt += 1
    cate_id = str(cate_id)[1:11]
    # Make Dir(./preview/train_XXXXXXXXX)
    category_dir = os.path.dirname("./preview/train_" + cate_id + "/")

    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

    # item count

    item_cnt = train_collection.find({'category_id': int(cate_id)}).count()

    for document in train_collection.find({'category_id': int(cate_id)}).limit(MAX_ITEM_CNT):
        img = document.get('imgs')[0]
        img_data = io.BytesIO(img.get('picture', None))
        image = imread(img_data)
        images.append(image)
        # print(type(image))

    print("[", category_ids_cnt, "][", cate_id, "]", item_cnt)
    image_augumentation(np.array(images), category_dir)
    # if category_ids_cnt > 1:
    #     break