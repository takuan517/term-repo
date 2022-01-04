import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.utils import np_utils

DATADIR = "./data"
CATEGORIES = ["human", "road"]
training_data = []

def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name),)
                training_data.append([img_array, class_num])

            except Exception as e:
                pass

create_training_data()

# random.shuffle(training_data)

X_train = []
y_train = []

# データセット作成
for feature, label in training_data:
    X_train.append(feature)
    y_train.append(label)

# numpy配列に変換
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.astype('float32')
X_train = X_train / 255.0

y_train = np_utils.to_categorical(y_train, 2)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=111)


# データセットの確認
# for i in range(0, 4):
#     print("学習データのラベル：", y_train[i])
#     plt.subplot(2, 2, i+1)
#     plt.axis('off')
#     plt.title(label = 'human' if y_train[i] == 0 else 'road')
#     img_array = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB)
#     plt.imshow(img_array)

# plt.show()

# ここからモデル作成
model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 144, 3)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100)
print(model.evaluate(X_test, y_test))
model.save("./data/human_model.h5")
