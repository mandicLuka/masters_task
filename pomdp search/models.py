import tensorflow as tf
from keras.layers import Dense, Conv2D, Reshape, Flatten, Input, Concatenate
from keras.models import Sequential, Model
import os
from yaml import safe_load as load_config
import keras.backend as K

def full_map_cnn(params):
    
    if bool(params["use_coords"]):
        num_channels = 7
    else:
        num_channels = 5
    grid = Input((params["num_rows"], params["num_cols"], num_channels))

    conv11 = Conv2D(filters=32, kernel_size=3, padding='same')(grid)
    conv12 = Conv2D(filters=64, kernel_size=5, padding='same')(grid)
    conv13 = Conv2D(filters=128, kernel_size=7, padding='same')(grid)
    concat1 = Concatenate()([conv11, conv12, conv13])
    conv2 = Conv2D(filters=128, kernel_size=5, padding='valid')(concat1)
    conv3 = Conv2D(filters=256, kernel_size=3, padding='valid')(conv2)
    conv4 = Conv2D(filters=256, kernel_size=3, padding='valid')(conv3)
    flat = Flatten()(conv4)
    fc1 = Dense(units=4096, activation="tanh")(flat)
    prediction = Dense(units=params["num_actions"])(fc1)

    full_map = Model(inputs=grid, outputs=prediction)
    return full_map
