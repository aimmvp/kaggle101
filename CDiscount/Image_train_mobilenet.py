import os
import time
import numpy as np

import keras
import mobilenet
import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

np.random.seed(777)


def make_model(img_len=None):
    model = mobilenet.MobileNet(input_shape=[160, 160, 3], classes=img_len)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.90), metrics=['accuracy'])

    tb_callback = keras.callbacks.TensorBoard(log_dir='../../CDiscount/log_graph')

    return model, tb_callback


training_path = "../../CDiscount/training/"
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.01, height_shift_range=0.01
                                   , shear_range=0.2 , zoom_range=[0.1, 0.3]
                                   )
train_generator = train_datagen.flow_from_directory(training_path, target_size=(160, 160), batch_size=64,
                                                    class_mode='binary')

# validation_path = "../../CDiscount/validation/"
# valid_datagen = ImageDataGenerator(rescale=1. / 255)
# valid_generator = valid_datagen.flow_from_directory(validation_path, target_size=(160, 160), batch_size=32,
#                                                     class_mode='binary')

training_list= os.listdir(training_path)

model, callbacks = make_model(img_len=len(training_list))

callback_best_only = ModelCheckpoint("../../CDiscount/checkpoint/weights_mobilenet.{epoch:02d}.h5", save_best_only=True, period=10)
# Found 40080 images belonging to 501 classes.
# Found 5010 images belonging to 501 classes.

# Found 40080 images belonging to 501 classes.
# Found 5010 images belonging to 501 classes.
# START TRAINING :  2017-12-30 20:27:23.425582

print("START TRAINING : ", datetime.datetime.now())
model.fit_generator(train_generator,
                    steps_per_epoch=155, # 1252
                    # validation_data= valid_generator,
                    # validation_steps=39, # 156
                    epochs=50,
                    callbacks=[callback_best_only, callbacks])

print("END TRAINING : ", datetime.datetime.now())
'''
Epoch 1/50
155/155 [==============================] - 2346s 15s/step - loss: 6.3356 - acc: 0.0026
Epoch 2/50
155/155 [==============================] - 2369s 15s/step - loss: 6.3382 - acc: 0.0017
Epoch 3/50
155/155 [==============================] - 2327s 15s/step - loss: 6.3361 - acc: 0.0018
Epoch 4/50
155/155 [==============================] - 2326s 15s/step - loss: 6.3377 - acc: 0.0021
Epoch 5/50
155/155 [==============================] - 2332s 15s/step - loss: 6.3417 - acc: 0.0027
'''
