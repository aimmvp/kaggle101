import pymongo
import pandas as pd
import os, io
import numpy as np
from pymongo import MongoClient
from scipy.misc import imread
from keras.preprocessing.image import ImageDataGenerator
import datetime


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


# Make category ids from category_names.csv
def make_category_ids_from_csv():
    category_names = pd.read_csv('input/category_names.csv', usecols=[0], dtype=int)
    category_ids = category_names.values
    return category_ids


image_gen = ImageDataGenerator(featurewise_center=False,
                               samplewise_center=False,
                               featurewise_std_normalization=False,
                               samplewise_std_normalization=False,
                               fill_mode='nearest')


def save_image_category_id(category_dir, image_ndarray):
    image_gen.fit(image_ndarray)

    idx = 1
    for _ in image_gen.flow(image_ndarray,
                            batch_size=len(image_ndarray),
                            save_to_dir=category_dir,
                            save_format='jpeg'):
        idx += 1
        if idx > 1:
            break


images = []
cnt = 0
print("START ", datetime.datetime.now(), "#############")
for cate_id in make_category_ids_from_csv():
    cate_id = str(cate_id)[1:11]
# cate_id = "1000016316"
    category_dir = os.path.dirname("../../train_img/train_" + cate_id + "/")

    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

    for document in train_collection.find({'category_id': int(cate_id)}).limit(MAX_ITEM_CNT):
        img = document.get('imgs')[0]
        img_data = io.BytesIO(img.get('picture', None))
        image = imread(img_data)
        images.append(image)

    save_image_category_id(category_dir, np.array(images))
    images.clear()
    cnt += 1
    if cnt % 100 == 0:
        print(datetime.datetime.now(), " // cnt : ", cnt)

print("END[", cnt, "]건 // ", datetime.datetime.now(), "#############")

'''
START  2017-12-19 22:40:21.000567 #############
2017-12-19 22:41:13.539740  // cnt :  100
2017-12-19 22:42:07.968937  // cnt :  200
2017-12-19 22:43:09.673377  // cnt :  300
2017-12-19 22:44:01.307194  // cnt :  400
2017-12-19 22:44:52.021721  // cnt :  500
2017-12-19 22:45:43.053423  // cnt :  600
2017-12-19 22:46:34.366261  // cnt :  700
2017-12-19 22:47:32.051783  // cnt :  800
2017-12-19 22:48:26.040209  // cnt :  900
2017-12-19 22:49:13.823490  // cnt :  1000
2017-12-19 22:50:04.757592  // cnt :  1100
2017-12-19 22:50:59.555754  // cnt :  1200
2017-12-19 22:51:50.340521  // cnt :  1300
2017-12-19 22:52:42.540680  // cnt :  1400
2017-12-19 22:53:35.542269  // cnt :  1500
2017-12-19 22:54:22.205425  // cnt :  1600
2017-12-19 22:55:14.393796  // cnt :  1700
2017-12-19 22:56:07.888532  // cnt :  1800
2017-12-19 22:56:53.455710  // cnt :  1900
2017-12-19 22:57:42.277024  // cnt :  2000
2017-12-19 22:58:31.365511  // cnt :  2100
2017-12-19 22:59:17.664667  // cnt :  2200
2017-12-19 23:00:05.597024  // cnt :  2300
2017-12-19 23:00:57.029497  // cnt :  2400
2017-12-19 23:01:53.756204  // cnt :  2500
2017-12-19 23:02:48.033300  // cnt :  2600
2017-12-19 23:03:31.150742  // cnt :  2700
2017-12-19 23:04:25.795895  // cnt :  2800
2017-12-19 23:05:16.176119  // cnt :  2900
2017-12-19 23:06:11.776127  // cnt :  3000
2017-12-19 23:16:48.500760  // cnt :  3100
2017-12-19 23:17:47.444118  // cnt :  3200
2017-12-19 23:18:39.797921  // cnt :  3300
2017-12-19 23:19:38.664612  // cnt :  3400
2017-12-19 23:20:58.701015  // cnt :  3500
2017-12-19 23:21:49.115317  // cnt :  3600
2017-12-19 23:22:40.241328  // cnt :  3700
2017-12-19 23:23:11.984668  // cnt :  3800
2017-12-19 23:24:00.931613  // cnt :  3900
2017-12-19 23:24:54.929789  // cnt :  4000
2017-12-19 23:25:41.356000  // cnt :  4100
2017-12-19 23:26:23.753060  // cnt :  4200
2017-12-19 23:27:34.187114  // cnt :  4300
2017-12-19 23:28:21.787325  // cnt :  4400
2017-12-19 23:29:10.068853  // cnt :  4500
2017-12-19 23:29:58.277588  // cnt :  4600
2017-12-19 23:30:38.554845  // cnt :  4700
2017-12-19 23:31:21.840653  // cnt :  4800
2017-12-19 23:32:02.408187  // cnt :  4900
2017-12-19 23:32:47.485364  // cnt :  5000
2017-12-19 23:33:36.227255  // cnt :  5100
2017-12-19 23:34:20.088880  // cnt :  5200
END[ 5270 ]건 //  2017-12-19 23:34:52.580519 #############
'''