"""
Defining a neural network using the Sequential API
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class SequentialModel:
    def __init__(self, n_output_nodes, input_shape):
        # Define the number of outputs
        self.__n_output_nodes = n_output_nodes
        # First define the model
        self.__model = Sequential()
        self.__model.add(tf.keras.Input(shape=input_shape))
        """TODO: Define a dense (fully connected) layer to compute z"""
        dense_layer = Dense(self.__n_output_nodes,
                            activation='sigmoid',
                            use_bias=True,
                            kernel_initializer=None, #'random ?
                            bias_initializer=None, #'random ?
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None)
        # Add the dense layer to the model
        self.__model.add(dense_layer)

    def call(self, x_input):
        """feed input into the model and predict the output!"""
        return self.__model.call(x_input)
