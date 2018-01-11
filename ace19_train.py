import os, time
import keras
import numpy as np
from scipy.misc import imread
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import mobilenet
import datetime

ACE19_BASE = "../ace19/"
bson_file_path = ACE19_BASE + "input"


def create_model(num_classes=None):
    model = mobilenet.MobileNet(input_shape=[160, 160, 3],
                                classes=num_classes)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.002, decay=1e-5, momentum=0.95, nesterov=True),
                  metrics=['accuracy'])

    tb_callback = keras.callbacks.TensorBoard(log_dir='./graph')
    return model, tb_callback


train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(ACE19_BASE + 'training',
                                                    target_size=(160, 160),
                                                    batch_size=64,
                                                    class_mode='binary')
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(ACE19_BASE + 'validation',
                                                    target_size=(160, 160),
                                                    batch_size=64,
                                                    class_mode='binary')

# Found 167988 images belonging to 300 classes.
# Found 20999 images belonging to 300 classes.

# 2018-01-06 22:24:32.415807:
# Found 24236 images belonging to 101 classes.
# Found 3029 images belonging to 101 classes.

# Found 3360 images belonging to 21 classes.
# Found 420 images belonging to 21 classes.


train_dirpath = ACE19_BASE + "training"
train_file_list = os.listdir(train_dirpath)
model, tb_callback = create_model(num_classes=len(train_file_list))

callback_best_only =  ModelCheckpoint('./models/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, period=2)

print("START====>", datetime.datetime.now())
model.fit_generator(train_generator,
                    steps_per_epoch=52,
                    validation_data=valid_generator,
                    validation_steps=6,
                    epochs=20,
                    callbacks=[callback_best_only, tb_callback])
print("END====>", datetime.datetime.now())