#!/usr/bin/env python
# -*- coding: utf-8 -*-

# http://www.mathgram.xyz/entry/keras/fcn
# https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D, GlobalAveragePooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras import backend as K
K.set_image_data_format('channels_last')

def simplecnn(nb_classes):
    model = Sequential()

    model.add(Conv2D(64,(3, 3),input_shape=(224, 224,3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(4,4)))

    model.add(Conv2D(128,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(4,4)))
   
    model.add(Conv2D(256,(3, 3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(nb_classes, activation='softmax'))

    return model

def cnn(nb_classes):
    model = Sequential()

    model.add(Conv2D(64,(3, 3),input_shape=(224, 224,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(4,4)))

    model.add(Conv2D(128,(3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(4,4)))
   
    model.add(Conv2D(256,(3, 3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(nb_classes, activation='softmax'))

    return model

def finetuneResNetV2(nb_classes):
    resnetV2 = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224, 224,3))
    for layer in resnetV2.layers[:400]:
        layer.trainable = False
    """
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnetV2.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))
    model = Model(input=resnetV2.input, output=top_model(resnetV2.output))
    """
    x = resnetV2.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=resnetV2.input, outputs=predictions)

    return model

def finetuneResNetV2_2(nb_classes):
    resnetV2 = InceptionResNetV2(include_top=False, weights=None, input_shape=(224, 224,3))
    #for layer in resnetV2.layers[:400]:
    #    layer.trainable = False
    """
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnetV2.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))
    model = Model(input=resnetV2.input, output=top_model(resnetV2.output))
    """
    x = resnetV2.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=resnetV2.input, outputs=predictions)

    return model

def finetuneVGG16(nb_classes):
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224,3))
    for layer in vgg16.layers[:3]: 
        layer.trainable = False
    """
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))
    model = Model(input=vgg16.input, output=top_model(vgg16.output))
    """
    x = vgg16.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=vgg16.input, outputs=predictions)

    return model

if( __name__ == '__main__' ):
	model = cnn()