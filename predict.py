# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import csv
from PIL import Image

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import load_img
from keras.utils import np_utils

import nn

def image_load(file_names):
    X = []

    for file_name in file_names:
        #im = Image.open( './train/'+img )
        im = load_img('./test/'+file_name, target_size=(224, 224))
        im = np.asarray( im, dtype = np.single ) / 255.
        X.append( im )

    return np.asarray(X)

nb_classes = 55
test_file = pd.read_csv("./sample_submit.tsv", header=None, delimiter="\t")

file_names = test_file[0]
X = image_load(file_names)

model = nn.cnn(nb_classes)

model.load_weights('imgs-fcn.hdf5')

print("load model")
Y = model.predict(X,batch_size=5)

Y = np.argmax(Y, axis = 1)

"""
result_array = []
for name, result in zip(file_names, map(str, Y)):
	result_array.append([name, result])

with open('result.tsv', 'w') as f:
	writer = csv.writer(f, lineterminator='\t')
	writer.writerows(result_array)

"""
df = pd.DataFrame({
		'name': pd.Series(file_names),
		'result': pd.Series(map(str, Y))
	})

df.to_csv("result.tsv", index=False, header=None, sep="\t")
