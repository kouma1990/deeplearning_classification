# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import nn

"""
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
K.set_session(sess)
"""

"""
def image_load(file_names):
    X_train = []

    for file_name in file_names:
        #im = Image.open( './train/'+img )
        im = load_img('./train/'+file_name, target_size=(224, 224))
        im = np.asarray( im, dtype = np.single ) / 255.
        X_train.append( im )

    return np.asarray(X_train)

nb_classes = 55
master_categories = pd.read_csv("./master.tsv", header=None, delimiter="\t")
train_file = pd.read_csv("./train_master.tsv", header=None, delimiter="\t")

file_names = train_file[0][1:1000]
Y_train = np.asarray(train_file[1][1:1000])
#file_names = train_file[0][1:]
#Y_train = np.asarray(train_file[1][1:])
Y_train = np_utils.to_categorical(Y_train, nb_classes)

X_train = image_load(file_names)
"""

nb_classes = 55
batch_size=32
epochs=100
seed = 1

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
    )
    

train_generator = train_datagen.flow_from_directory(
    './sorted_train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed,
    classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", 
             "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31" , "32", "33", "34", "35", "36", "37", "38", "39", "40", "41" , "42", "43", "44", "45", "46", "47", "48", "49", 
             "50", "51" , "52", "53", "54"])

model = nn.finetuneResNetV2_2(nb_classes)
model.summary()

#model.load_weights('imgs-fcn.hdf5')
"""
opt = Adam(decay=1E-4) #lr *= (1. / (1. + self.decay * self.iterations))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    nb_epoch=200)
"""

#opt = Adam(decay=1E-4) #lr *= (1. / (1. + self.decay * self.iterations))
opt = SGD(lr=.001, momentum=.9)
#opt = RMSprop(lr=.0045, decay=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

#model.fit( X_train, Y_train, batch_size=batch_size, epochs=50, verbose=1 )
model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=epochs)

model.save('imgs-fcn2.hdf5')

#print(history)
#print(Y.shape)

# pydot之関係で使えない
#from keras.utils import plot_model
#plot_model(model, to_file="model.png", show_shapes=True)






