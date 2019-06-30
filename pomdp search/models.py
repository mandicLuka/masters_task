import tensorflow as tf
from tensorflow.keras import layers, Model

import os
from yaml import safe_load as load_config

def full_map_cnn(params):
    
    if bool(params["use_coords"]):
        num_channels = 7
    else:
        num_channels = 5
    grid = layers.Input((params["num_rows"], params["num_cols"], num_channels))

    conv11 = layers.Conv2D(filters=32, kernel_size=3, padding='same')(grid)
    conv12 = layers.Conv2D(filters=64, kernel_size=5, padding='same')(grid)
    conv13 = layers.Conv2D(filters=128, kernel_size=7, padding='same')(grid)
    concat1 = layers.Concatenate()([conv11, conv12, conv13])
    conv2 = layers.Conv2D(filters=128, kernel_size=5, padding='valid')(concat1)
    conv3 = layers.Conv2D(filters=256, kernel_size=3, padding='valid')(conv2)
    conv4 = layers.Conv2D(filters=256, kernel_size=3, padding='valid')(conv3)
    flat = layers.Flatten()(conv4)
    fc1 = layers.Dense(units=4096, activation="tanh")(flat)
    prediction = layers.Dense(units=params["num_actions"])(fc1)

    full_map = Model(inputs=grid, outputs=prediction)
    return full_map

class FullMapCNN(Model):

    def __init__(self, params):
        super(FullMapCNN, self).__init__()
        self.params = params
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.build_model()
        self.sess.run(self.init)

    def build_model(self):
        if bool(self.params["use_coords"]):
            num_channels = 7
        else:
            num_channels = 5
        shape = (self.params["num_rows"], self.params["num_cols"], num_channels)
        with self.graph.as_default():
            self.X = tf.placeholder("float", [None, *shape])
            self.y = tf.placeholder("float", [None, self.params["num_actions"]])

            conv11 = layers.Conv2D(filters=32, kernel_size=3, padding='same')(self.X)
            conv12 = layers.Conv2D(filters=64, kernel_size=5, padding='same')(self.X)
            conv13 = layers.Conv2D(filters=128, kernel_size=7, padding='same')(self.X)
            concat1 = layers.Concatenate()([conv11, conv12, conv13])
            conv2 = layers.Conv2D(filters=128, kernel_size=5, padding='valid')(concat1)
            conv3 = layers.Conv2D(filters=256, kernel_size=3, padding='valid')(conv2)
            conv4 = layers.Conv2D(filters=256, kernel_size=3, padding='valid')(conv3)
            flat = layers.Flatten()(conv4)
            fc1 = layers.Dense(units=4096, activation="tanh")(flat)
            self.prediction = layers.Dense(units=self.params["num_actions"])(fc1)

            self.loss = tf.losses.mean_squared_error(self.y, self.prediction)
            self.optimizer = tf.train.AdamOptimizer(self.params["learning_rate"]).minimize(self.loss)
            self.init = tf.global_variables_initializer()

    def train(self, X, y):
        print("Train on " + str(X.shape[0]) + " samples.")
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: X, self.y: y})
        print("Loss: " + str(loss))

    def predict(self, X):
        prediction = self.sess.run(self.prediction, feed_dict={self.X: X})
        return prediction

    def set_trainable_weights(self, weights):
        w = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i, value in enumerate(weights):
            self.sess.run(w[i].assign(value))

    def get_trainable_weights(self):
        values = []
        variables = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in variables: 
            values.append(self.sess.run(var))
        return values