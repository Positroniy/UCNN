# import libraries
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D
from keras.regularizers import l2
import numpy as np
import os
import matplotlib.pyplot as plt

def Model_1(x_train, num_classes,params):
    # +Cifar 10 with dropout withou batchNormalize
    # # model architecture
    model = Sequential()

    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size']), padding='same', input_shape=x_train.shape[1:]))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(params['d_lay']))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def Model_2(x_train, num_classes,params):
    model = Sequential()
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size']),
                 padding='same',
                 input_shape=x_train.shape[1:]))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))

    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))

    model.add(Flatten())
    model.add(Dense(params['d_lay']))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
def Model_3(x_train, num_classes,params):
    model = Sequential()
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size']), padding='same',
                 input_shape=x_train.shape[1:]))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(params['d_lay']))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
def Model_4(x_train, num_classes,params):
    model = Sequential()
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size']), padding='same', input_shape=x_train.shape[1:]))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))

    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))

    model.add(Flatten())
    model.add(Dense(params['d_lay'], kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
def Model_5(x_train, num_classes,params):
    model = Sequential()
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size']), padding='same',
                input_shape=x_train.shape[1:]))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(params['d_lay'],kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
def Model_6(x_train, num_classes,params):
    model = Sequential()
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size']), padding='same',
                 input_shape=x_train.shape[1:]))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_2'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Conv2D(params['num_filters_3'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_3'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(params['d_lay'],kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
def Model_7(x_train, num_classes,params):
    model = Sequential()
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size']), padding='same',
                 input_shape=x_train.shape[1:]))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size']), padding='same'))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(Conv2D(params['num_filters_1'], (params['size'], params['size'])))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(params['p_size'], params['p_size'])))
    model.add(Dropout(params['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(params['d_lay'], kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=params['L_ReLU_alpha']))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def build_model(modelx,x_train,num_classes,params):
    dic_models = {
                'Model_1': Model_1(x_train,num_classes,params),
                'Model_2': Model_2(x_train,num_classes,params),
                'Model_3': Model_3(x_train,num_classes,params),
                'Model_4': Model_4(x_train,num_classes,params),
                'Model_5': Model_5(x_train,num_classes,params),
                'Model_6': Model_6(x_train,num_classes,params),
                'Model_7': Model_7(x_train,num_classes,params),
                }
    model = dic_models[modelx]
    # summary
    print(model.summary())
    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=params['optimizer'],
                  metrics=['accuracy']) 
    return model
