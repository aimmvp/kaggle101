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

4. ace19_make_file.py
  - save image files at file system per training / validation / testing and category_id
  - autumentation images by ImageDataGenerator
'''


import io, os
import bson
import numpy as np
from scipy.misc import imread
from keras.preprocessing.image import ImageDataGenerator


ACE19_BASE = "../../ace19/"
bson_file_path = ACE19_BASE + "input"
bson_list = os.listdir(bson_file_path)


image_datagen = ImageDataGenerator(featurewise_center=True,
                                   samplewise_center=True,
                                   featurewise_std_normalization=True,
                                   samplewise_std_normalization=True,
                                   fill_mode='nearest')


def get_image_array(documents):
    images = []
    img_idx = 0;

    for doc in documents:
        img = doc.get('imgs')[0]
        img_data = io.BytesIO(img.get('picture', None))
        im = imread(img_data)
        im = im.astype('float32') / 255.0

        images.append(im)
        img_idx += 1
        if img_idx > 350:
            break

    num_of_total = len(images)
    num_of_80, num_of_10 = int(num_of_total * (8 / 10)), int(num_of_total * (1 / 10))

    return np.array(images[0:num_of_80]), np.array(images[num_of_80:num_of_80+num_of_10]), np.array(images[num_of_80+num_of_10:])


def image_augu(img_nparr, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_cnt = 0
    image_datagen.fit(img_nparr)

    for _ in image_datagen.flow(img_nparr,
                                batch_size=len(img_nparr),
                                save_to_dir=img_dir,
                                save_format='jpeg'):
        img_cnt += 1
        if img_cnt > 1:
            break


for bson_name in bson_list:
    if bson_name.find('train_') is -1:
        continue

    documents = bson.decode_file_iter(open(bson_file_path + '/' + bson_name, 'rb'))
    categories = [(doc['_id'], doc['category_id']) for doc in documents]
    category_id = categories[0][1]

    training_dir = os.path.dirname(ACE19_BASE + "training/" + str(category_id) + "/")
    validation_dir = os.path.dirname(ACE19_BASE + "validation/" + str(category_id) + "/")
    testing_dir = os.path.dirname(ACE19_BASE + "testing/" + str(category_id) + "/")

    if os.path.exists(training_dir):
        continue
    if os.path.exists(validation_dir):
        continue
    if os.path.exists(testing_dir):
        continue

    train, validate, test = get_image_array(bson.decode_file_iter(open(bson_file_path + '/' + bson_name, 'rb')))
    print("[", bson_name, "]")
    if len(train) != 0 :
        image_augu(train, training_dir)
        image_augu(validate, validation_dir)
        image_augu(test, testing_dir)
