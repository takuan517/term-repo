import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

X_train = []
y_train = []

for picture in list_pictures('./data/human'):
    img = img_to_array(load_img(picture, target_size=(256,144)))
    X_train.append(img)

    y_train.append(0)

for picture in list_pictures('./data/road'):
    img = img_to_array(load_img(picture, target_size=(256,144)))
    X_train.append(img)

    y_train.append(1)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.astype('float32')
X_train = X_train / 255.0

y_train = np_utils.to_categorical(y_train, 2)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=111)


model = Sequential()

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
# model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100)
print(model.evaluate(X_test, y_test))
model.save("./data/human_model5.h5")
