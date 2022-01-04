from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import time

start = time.time()
jpg_name = './data/test/001.jpg'
my_model = './data/human_model5.h5'

model = load_model(my_model)

img_path = (jpg_name)
img = img_to_array(load_img(img_path, target_size = (256,144)))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]

label=['human','road']
pred = model.predict(img_nad, batch_size=1, verbose=0)
score = np.max(pred)
pred_label = label[np.argmax(pred[0])]
print('name:',pred_label)
print('score:',score)
process_time = time.time() - start
print(process_time)
