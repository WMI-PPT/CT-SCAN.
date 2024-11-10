"""
Created on Sun Nov 3 21:08:01 2024

@author: WANG YIFAN
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical, plot_model
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import keras
import cv2
import os
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Concatenate


def create_model():
    # Define three inputs
    input1 = Input(shape=(50, 50, 3), name='input1')
    input2 = Input(shape=(50, 50, 3), name='input2')
    input3 = Input(shape=(50, 50, 3), name='input3')

    def create_branch(input_layer):
        x = Conv2D(32, (3, 3), padding='same')(input_layer)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(48, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(48, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        return x

    # Create branches
    branch1 = create_branch(input1)
    branch2 = create_branch(input2)
    branch3 = create_branch(input3)

    # Output layers
    category_predict1 = Dense(100, activation='relu', name='category_predict1')(branch1)
    category_predict2 = Dense(100, activation='relu', name='category_predict2')(branch2)
    category_predict3 = Dense(100, activation='relu', name='category_predict3')(branch3)

    # Merge the three branches
    merge = Concatenate()([category_predict1, category_predict2, category_predict3])

    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(merge)  # Binary classification with 1 output neuron

    # Create model
    model = Model(inputs=[input1, input2, input3], outputs=[output])

    # Set callbacks
    callbacks = [keras.callbacks.TensorBoard(log_dir='my_log_dir'),
                 keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)]

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001, decay=0.01),
                  loss='binary_crossentropy',  # Use binary crossentropy loss function
                  metrics=['accuracy'])

    return model, callbacks


# Call the function and create the model
model, callbacks = create_model()
model.summary()
