from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

my_model = './data/human_model5.h5'

model = load_model(my_model)

i = 0
while True:
    jpg_name = "./data/test/%d.jpg"%i
    i += 1
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
    if pred_label == 'human':
        break
        # serverへリクエスト投げる
        #　if response == alertとかだったらwhileを止める break
