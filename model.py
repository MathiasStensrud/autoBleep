from keras.layers import Activation, Conv2D, Dense,Flatten, TimeDistributed, Dropout, Bidirectional, MaxPooling2D
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import glob
import cv2 as cv
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam

import numpy as np
callback = ModelCheckpoint('./logs',save_best_only=True)

def change_trainable_layers(model, trainable_index):

    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

# def create_transfer_model(input_size, 2, weights = 'imagenet'):
#
#         base_model = VGG16(weights=weights,
#                           include_top=False,
#                           input_shape=input_size)
#
#         model = base_model.output
#
#         return model


def create_model(input_size, n_categories):

    optimizer=Adam(lr=0.0001)
    base_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=input_size)

    tl = base_model.output
    predictions = Flatten()(tl)
    tl = Model(inputs=base_model.input, outputs=predictions)
    for layer in tl.layers[:-3]:
        layer.trainable = False

    nb_filters = 4
    kernel_size = (5, 5)
    pool_size=(2,2)
    cnn = Sequential()
    # 2 convolutional layers followed by a pooling layer followed by dropout
    cnn.add(Conv2D(nb_filters, kernel_size,
                            padding='valid',
                            input_shape=input_size))
    cnn.add(MaxPooling2D(pool_size=pool_size))
    cnn.add(Dropout(0.2))
    cnn.add(Conv2D(nb_filters, kernel_size))
    cnn.add(MaxPooling2D(pool_size=pool_size))
    cnn.add(Dropout(0.2))
    cnn.add(Conv2D(nb_filters, kernel_size))
    cnn.add(MaxPooling2D(pool_size=pool_size))
    cnn.add(Dropout(0.2))
    cnn.add(Flatten())
    # model.add(LSTM(()))
    model= Sequential()
    model.add(TimeDistributed(tl, input_shape=[6,140,140,3]))
    model.add(Bidirectional(LSTM(1)))
    model.add(Dense(n_categories))
    model.add(Activation('sigmoid'))
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # model.add(Convolution2D(nb_filters, kernel_size))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))
    # # transition to an mlp
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_categories))
    # model.add(Activation('softmax'))
    return model

def build_data():
    filenames = glob.glob("data/pos/*.jpg")
    filenames.sort()
    images = [cv.imread(img) for img in filenames]
    dat=[]
    y=[]
    i=0
    while i < len(images):
        dat.append(images[i:i+6])
        y.append([1,0])
        i+=6
    filenames = glob.glob("data/neg/*.jpg")
    filenames.sort()
    images = [cv.imread(img) for img in filenames]
    i=0
    while i < len(images):
        dat.append(images[i:i+6])
        y.append([0,1])
        i+=6
    dat=np.array(dat)
    y=np.array(y)
    y.reshape(-1,1)
    return(dat, y)

def build_validation():
    filenames = glob.glob("data/valPos/*.jpg")
    filenames.sort()
    images = [cv.imread(img) for img in filenames]
    dat=[]
    y=[]
    i=0
    while i < len(images):
        dat.append(images[i:i+6])
        y.append([1,0])
        i+=6
    filenames = glob.glob("data/valNeg/*.jpg")
    filenames.sort()
    images = [cv.imread(img) for img in filenames]
    i=0
    while i < len(images):
        dat.append(images[i:i+6])
        y.append([0,1])
        i+=6
    dat=np.array(dat)
    y=np.array(y)
    y.reshape(-1,1)
    return(dat, y)

if __name__=='__main__':
    size=[140,140,3]
    categories=1
