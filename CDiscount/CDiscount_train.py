import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout

np.random.seed(555)

# 1. Make Dataset
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=10,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.7,
                                  zoom_range=[0.9, 2.2],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
                    '../../train_img', target_size=(180,180), batch_size=32,
                    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    '../../test_img',
    target_size=(180, 180),
    batch_size=32,
    # class_mode='categorical'
    class_mode='binary'
)

# 2. Make Model
model = Sequential()
print("MODEL STEP 1")
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(180, 180, 3)))

print("MODEL STEP 2")

# model.add(Conv2D(64, (3,3), activation='relu'))
# print("MODEL STEP 3")
# model.add(MaxPooling2D(pool_size=(2,2)))
# print("MODEL STEP 4")
# model.add(Flatten())
# print("MODEL STEP 5")
# model.add(Dense(128, activation='relu'))
# print("MODEL STEP 6")
# model.add(Dense(3, activation='softmax'))
# print("MODEL STEP 7")

# 3. 모델 학습과정 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Traning Model
model.fit_generator(train_generator,
                    steps_per_epoch=15 * 100,
                    epochs=20,
                    validation_data=test_generator,
                    validation_steps=5)

# 5. Evaluate Model
print("##### Evaluate #######")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1] * 100))

# 6. Use Model
print("##### Predict #####")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0.03f}".format(x)})
print(test_generator.class_indices)
print(output)


