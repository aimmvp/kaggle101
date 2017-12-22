import os
import time
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten
)
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

np.random.seed(777)

train_datagen = ImageDataGenerator(rescale=1./255           # Rescaling Factor before any other transformation.
                                   , rotation_range=10      # 지정된 각도 범위 내에서 임의로 원본 이미지 회전 ( 0 ~ 10도)
                                   , width_shift_range=0.01  # 수평방향 이동 범위 width * 0.01 = 180 * 0.01 =  1.8
                                   , height_shift_range=0.01 # 수직방향 이동 범위 height * 0.01 = 180 * 0.01 = 1.8
                                   , shear_range=0.2         # 시계 반대 방향 밀림 강도 단위 라디안
                                   , zoom_range=[0.1, 0.3]  # 확대 축소 범위 : 1-수치 ~ 1+수치
                                   # , horizontal_flip=True   # 수평방향 뒤집기
                                   # , vertical_flip=True     # 수직방향 뒤집기
                                   # , fill_mode='nearest'    # default : nearest
                                   )

# * flow_from_directory
# Takes the path to a directory, and generates batches of augmented/normalized data.
# Yields batches indefinitely, in an infinite loop.
train_generator = train_datagen.flow_from_directory('../../train_img',
                                                    target_size=(180, 180),
                                                    batch_size=32,      # default 32
                                                    class_mode='binary')

train_img_list= os.listdir("../../train_img")

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=[180, 180, 3]))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(train_img_list), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# callback_tb = keras.callbacks.TensorBoard(
#     log_dir='./logs/{}-{}'.format(time.time(), ' '.join(l.name for l in model.layers))
# )
callback_best = ModelCheckpoint("./image_train_best.h5", save_best_only=True)
model.fit_generator(train_generator, steps_per_epoch=1, epochs=10, callbacks=[callback_best])

