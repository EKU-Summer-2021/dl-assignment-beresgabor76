import unittest
import numpy as np
import tensorflow as tf
from src import OurDenseLayer


class OurDenseLayerTest(unittest.TestCase):

    def test_our_dense_layer(self):
        tf.random.set_seed(1)
        layer = OurDenseLayer(3)
        layer.build((1, 2))
        x_input = tf.constant([[1, 2.]], shape=(1, 2))
        y = layer.call(x_input)
        assert tf.rank(y).numpy() == 2, "matrix must be of rank 2"
        assert tf.shape(y).numpy().tolist() == [1, 3], "matrix is incorrect shape"
        np.allclose([[0.26978594, 0.45750415, 0.66536945]], y.numpy())


if __name__ == '__main__':
    unittest.main()
