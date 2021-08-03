"""
Defining a network Layer
# n_output_nodes: number of output nodes
# input_shape: shape of the input
# x: input to the layer
"""
import tensorflow as tf


class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes
        self.W = None
        self.b = None

    def build(self, input_shape):
        d = int(input_shape[-1])
        # Define and initialize parameters: a weight matrix W and bias b
        # Note that parameter initialization is random!
        self.W = self.add_weight("weight", shape=[d, self.n_output_nodes]) # note the dimensionality
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes]) # note the dimensionality

    def call(self, x):
        """"define the operation for z (hint: use tf.matmul)"""
        z = tf.add(tf.matmul(x, self.W), self.b)
        """define the operation for out (hint: use tf.sigmoid)"""
        y = tf.sigmoid(z)
        return y
