"""
Module for defining a model using subclassing and specifying custom behavior
"""
from tensorflow.keras import Model
import tensorflow as tf


class IdentityModel(Model):
    """
    Defining a model using subclassing and specifying custom behavior
    """
    # As before, in __init__ we define the Model's layers
    # Since our desired behavior involves the forward pass, this part is unchanged
    def __init__(self, n_output_nodes):
        super().__init__()
        self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

    def call(self, inputs, isidentity=False):
        """Implement the behavior where the network outputs the input,
        unchanged, under control of the isidentity argument."""
        x = self.dense_layer(inputs)
        #Implement identity behavior
        if isidentity:
            return inputs
        return x

    def get_config(self):
        """Abstract method overridden"""
