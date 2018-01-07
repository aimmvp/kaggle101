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

 2. CDiscount/ace19_split_collection.py
  - Make index for train.bson
  ```
  train_collection.create_index("category_id")
  ```
  - split collection per category_id only 100 collection
    ( 200 ~ 300 개의 item 을 가지고 있는 카테고리 아이디 사용)

 3. CDiscount/ace19_make_bson.py
  - Make bson file at file system per category_id 

 4. CDiscount/ace19_make_file.py
  - save image files at file system per training / validation /testing  
  - augumentation images by ImageDataGenerator only 350 images

 5. CDiscount/ace19_train.py
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
