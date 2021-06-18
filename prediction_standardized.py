import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os


folder = '/home/a/Documents/recognition_of_my_handwritten_digits/img_st/'
files = [f for f in os.listdir(folder) if 'png' in f or 'jpg' in f]


# model = load_model('straight_bs200e30.h5')
model = load_model('es.h5')


def predict(folder, file):
    img = cv2.imread(folder+file, cv2.IMREAD_GRAYSCALE)
    im = img.reshape(1, 28, 28, 1)/255
    return np.argmax(model.predict(im))


count = 0
for i in range(0, len(files)):
    if int(files[i][0]) == predict(folder, files[i]):
        count += 1


print(count / len(files) * 100, '%')


for f in files:
    print(f, predict(folder, f), sep='  ')


count = 0
for i in range(0, len(files)):
    if int(files[i][0]) == predict(folder, files[i]):
        count += 1
