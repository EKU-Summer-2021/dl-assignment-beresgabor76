"""
Module for defining a model using subclassing
"""

from tensorflow.keras import Model
import tensorflow as tf


class SubclassModel(Model):
    """
    Defining a model using subclassing
    """
    # In __init__, we define the Model's layers
    def __init__(self, n_output_nodes):
        super().__init__()
        # Our model consists of a single Dense layer. Define this layer.
        self.dense_layer = tf.keras.layers.Dense(n_output_nodes,
                                                 activation='sigmoid',
                                                 use_bias=True,
                                                 kernel_initializer=None,
                                                 bias_initializer=None,
                                                 kernel_regularizer=None,
                                                 bias_regularizer=None,
                                                 activity_regularizer=None,
                                                 kernel_constraint=None,
                                                 bias_constraint=None)

    # In the call function, we define the Model's forward pass.
    def call(self, inputs, training=None, mask=None):
        return self.dense_layer(inputs)

    def get_config(self):
        """Abstract method overridden"""
