"""
Module for defining a network Layer
# n_output_nodes: number of output nodes
# input_shape: shape of the input
# x: input to the layer
"""
import tensorflow as tf


class OurDenseLayer(tf.keras.layers.Layer):
    """
    Defining a network Layer
    """
    def __init__(self, n_output_nodes):
        super().__init__()
        self.n_output_nodes = n_output_nodes
        self.weight = None
        self.bias = None

    def build(self, input_shape):
        dim = int(input_shape[-1])
        # Define and initialize parameters: a weight matrix W and bias b
        # Note that parameter initialization is random!
        self.weight = self.add_weight("weight", shape=[dim, self.n_output_nodes]) # note the dimensionality
        self.bias = self.add_weight("bias", shape=[1, self.n_output_nodes]) # note the dimensionality

    def call(self, x_input):
        # define the operation for z (hint: use tf.matmul)
        z_out = tf.add(tf.matmul(x_input, self.weight), self.bias)
        # define the operation for out (hint: use tf.sigmoid)
        y_out = tf.sigmoid(z_out)
        return y_out
