import os, io
import bson
import numpy as np
from scipy.misc import imread
from keras.preprocessing.image import ImageDataGenerator
import datetime

dataset_file_path_base = '../../CDiscount/input/'

bson_file_list = os.listdir(dataset_file_path_base)
bson_file_list.sort()


image_gen = ImageDataGenerator(featurewise_center=True,
                                    samplewise_center=True,
                                    featurewise_std_normalization=True,
                                    samplewise_std_normalization=True,
                                    fill_mode='nearest')
MAX_IMAGE_CNT = 100


def image_augumentation(image_ndarray, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    image_gen.fit(image_ndarray)

    idx = 1
    for _ in image_gen.flow(image_ndarray,
                            batch_size=len(image_ndarray),
                            save_to_dir=image_path,
                            save_format='jpeg'):
        idx += 1
        if idx > 1:
            break


def get_image_np_array(documents):
    images = []
    image_cnt = 0

    for document in documents:
        img = document.get('imgs')[0]
        img_data = io.BytesIO(img.get('picture', None))
        image = imread(img_data)
        image = image.astype('float32') / 255.0
        images.append(image)
        image_cnt += 1

        if image_cnt > MAX_IMAGE_CNT:
            break

    total_cnt = len(images)
    num_of_80 = int(total_cnt * (8 / 10))
    num_of_10 = int(total_cnt * (1 / 10))
    # print(total_cnt, "/", num_of_80, "/", num_of_10)

    return np.array(images[0:num_of_80]), np.array(images[num_of_80:num_of_80+num_of_10]), np.array(images[num_of_80+num_of_10:])


bson_file_path = "";

print("######  START ", datetime.datetime.now())
for bson_file in bson_file_list:
    if bson_file.find('train_') is -1:
        continue

    # doc = bson.decode_file_iter(open(dataset_file_path_base + bson_file, 'rb'))
    category_id = bson_file.split('_')[1].split('.')[0]

    training_path = "../../CDiscount/training/" + category_id + "/"
    validation_path = "../../CDiscount/validation/" + category_id + "/"
    testing_path = "../../CDiscount/testing/" + category_id + "/"

    # if os.path.exists(training_path) or os.path.exists(validation_path) or os.path.exists(testing_path):
    #     continue

    bson_file_path = dataset_file_path_base + bson_file
    # print("bson_file_path : ", bson_file_path)
    train_arr, validation_arr, testing_arr = get_image_np_array(bson.decode_file_iter(open(bson_file_path, 'rb')))
    # print("t/", train_arr.shape)
    # print("v/",validation_arr.shape)
    # print("e/", testing_arr.shape)
    image_augumentation(train_arr, training_path)
    image_augumentation(validation_arr, validation_path)
    image_augumentation(testing_arr, testing_path)

print("###### END ", datetime.datetime.now())

