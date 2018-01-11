### Kaggle Competition : Cdiscount's Image Classification Challenge
 - Categorize e-commerce photos
 - URL : https://www.kaggle.com/c/cdiscount-image-classification-challenge

### ace19_dev 참고
 - https://github.com/ace19-dev/Categorize-ecommerce-photos

### 실행 순서
 1. bson 다운로드 & Restore
  - Kaggle > Data 탭에서 train.bson 파일 다운로드(https://www.kaggle.com/c/cdiscount-image-classification-challenge/data)
  - bson 파일 저장위치로 이동
  ```
  $ cd ~/tensorflow/kaggle_master/Cdiscount
  ```
  - MongoDB 기동
  ```
  $ mongod
  ```
  - bson 파일 restore(https://docs.mongodb.com/manual/reference/program/mongorestore/)
  ```
  $ mongorestore --drop -d train_10_5 -c train train.bson
  # --drop : target database 에서 collection 을 drop 시킨다.
  # --d : database 명
  # --c : collection 명
  ```

 2. ace19_split_collection.py
  - Make index for train.bson
  ```
  train_collection.create_index("category_id")
  ```
  - split collection per category_id only 100 collection
    ( 200 ~ 300 개의 item 을 가지고 있는 카테고리 아이디 사용)

 3. ace19_make_bson.py
  - Make bson file at file system per category_id 

 4. ace19_make_file.py
  - save image files at file system per training(80%) / validation(10%) /testing(10%)  
  - augumentation images by ImageDataGenerator only 350 images

 5. ace19_train.py
  - training model(mobilenet)

```
Found 24236 images belonging to 101 classes.
Found 3029 images belonging to 101 classes.
2018-01-06 22:24:32.415807: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Epoch 1/50
1000/1000 [==============================] - 15404s 15s/step - loss: 3.4150 - acc: 0.1877 - val_loss: 4.8928 - val_acc: 0.0868
Epoch 2/50
1000/1000 [==============================] - 15315s 15s/step - loss: 2.2050 - acc: 0.4155 - val_loss: 8.1151 - val_acc: 0.0485
1000/1000 [==============================] - 15362s 15s/step - loss: 1.7068 - acc: 0.5310 - val_loss: 5.8276 - val_acc: 0.1463
Epoch 4/50[==============>...............] - ETA: 2:05:42 - loss: 1.8065 - acc: 0.5063
1000/1000 [==============================] - 15256s 15s/step - loss: 1.3497 - acc: 0.6168 - val_loss: 8.9592 - val_acc: 0.0854
Epoch 5/50
1000/1000 [==============================] - 15264s 15s/step - loss: 1.0713 - acc: 0.6916 - val_loss: 7.4099 - val_acc: 0.0864
```
```python
model.fit_generator(train_generator,
                    steps_per_epoch=52,
                    validation_data=valid_generator,
                    validation_steps=6,
                    epochs=20,
                    callbacks=[callback_best_only, tb_callback])
```

```
Epoch 1/20
52/52 [==============================] - 825s 16s/step - loss: 3.0409 - acc: 0.0685 - val_loss: 3.0496 - val_acc: 0.0495
Epoch 2/20
52/52 [==============================] - 821s 16s/step - loss: 2.7974 - acc: 0.1563 - val_loss: 3.0700 - val_acc: 0.0417
Epoch 3/20
52/52 [==============================] - 820s 16s/step - loss: 2.4629 - acc: 0.2506 - val_loss: 3.1324 - val_acc: 0.0417
Epoch 4/20
52/52 [==============================] - 819s 16s/step - loss: 2.2284 - acc: 0.3125 - val_loss: 3.2254 - val_acc: 0.0495
Epoch 5/20
52/52 [==============================] - 818s 16s/step - loss: 2.0327 - acc: 0.3651 - val_loss: 3.3169 - val_acc: 0.0417
Epoch 6/20
52/52 [==============================] - 817s 16s/step - loss: 1.8965 - acc: 0.4048 - val_loss: 3.4151 - val_acc: 0.0417
Epoch 7/20
52/52 [==============================] - 825s 16s/step - loss: 1.7680 - acc: 0.4507 - val_loss: 3.5311 - val_acc: 0.0521
Epoch 8/20
52/52 [==============================] - 810s 16s/step - loss: 1.5939 - acc: 0.5012 - val_loss: 3.5553 - val_acc: 0.0417
Epoch 9/20
52/52 [==============================] - 815s 16s/step - loss: 1.4426 - acc: 0.5526 - val_loss: 3.7703 - val_acc: 0.0443
Epoch 10/20
52/52 [==============================] - 822s 16s/step - loss: 1.3721 - acc: 0.5655 - val_loss: 4.2210 - val_acc: 0.0755
Epoch 11/20
52/52 [==============================] - 819s 16s/step - loss: 1.1837 - acc: 0.6250 - val_loss: 3.3885 - val_acc: 0.1562
Epoch 12/20
52/52 [==============================] - 818s 16s/step - loss: 1.1315 - acc: 0.6421 - val_loss: 3.4546 - val_acc: 0.1510
Epoch 13/20
52/52 [==============================] - 826s 16s/step - loss: 1.0112 - acc: 0.6785 - val_loss: 4.1653 - val_acc: 0.1484
Epoch 14/20
52/52 [==============================] - 809s 16s/step - loss: 0.8825 - acc: 0.7224 - val_loss: 4.5479 - val_acc: 0.2214
Epoch 15/20
52/52 [==============================] - 815s 16s/step - loss: 0.8210 - acc: 0.7392 - val_loss: 4.2705 - val_acc: 0.1823
Epoch 16/20
52/52 [==============================] - 815s 16s/step - loss: 0.7683 - acc: 0.7572 - val_loss: 5.0170 - val_acc: 0.1354
Epoch 17/20
52/52 [==============================] - 831s 16s/step - loss: 0.7509 - acc: 0.7593 - val_loss: 5.4447 - val_acc: 0.1250
Epoch 18/20
52/52 [==============================] - 825s 16s/step - loss: 0.6557 - acc: 0.7894 - val_loss: 6.3842 - val_acc: 0.1094
Epoch 19/20
52/52 [==============================] - 818s 16s/step - loss: 0.6087 - acc: 0.8074 - val_loss: 5.5194 - val_acc: 0.1667
Epoch 20/20
52/52 [==============================] - 827s 16s/step - loss: 0.5317 - acc: 0.8305 - val_loss: 6.0356 - val_acc: 0.1276
```
