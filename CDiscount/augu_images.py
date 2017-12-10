import bson, io
from scipy.misc import imread
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd

# 참고 : https://tykimos.github.io/2017/06/10/CNN_Data_Augmentation/
# with open("./input/under_list.csv", "r") as f:

# Make File Name from under_list.csv ( category_id, item_cnt)
file_name = "./input/train_1000022325.bson"
item_cnt = 12
MAX_ITEM_CNT = 100

### Image Augumentation

# Fix Random Seed
np.random.seed(777)

# Make Dataset
data_aug_gen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=10,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.7,
                                  zoom_range=[0.9, 2.2],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

def get_features_label(documents):
    images = []
    labels = []
    for docu in documents:
        category_id = docu.get('category_id', '')
        img = docu.get('imgs')[0]
        data = io.BytesIO(img.get('picture', None))
        im = imread(data)

        if category_id:
            label = category_id
        else:
            label = None

        labels.append(label)
        images.append(im)

    return np.array(images), np.array(labels)


def image_augumentation(X_train, y_train):
    i = 1
    for batch in data_aug_gen.flow(X_train, y_train,
                                   batch_size=1,
                                   save_to_dir='output/preview',
                                   save_prefix='',
                                   save_format='png'):
        i += 1
        if i >= MAX_ITEM_CNT - len(y_train):
            break


X_train, y_train = get_features_label(bson.decode_file_iter(open(file_name, 'rb')))
print(X_train)
image_augumentation(X_train, y_train)

'''
Data Sample
{
    "_id" : 1,
    "imgs": [
        {
            "picture": BinData(......................)
        }
    ],
    "category_id" : 1000010653
}
'''
