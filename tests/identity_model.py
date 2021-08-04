import unittest

import numpy as np
import tensorflow as tf
from src import IdentityModel


class IdentityModelTest(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(1)
        n_output_nodes = 3
        self.model = IdentityModel(n_output_nodes)

    def test_identity_model_when_isidentity(self):
        x_input = tf.constant([[1, 2.]], shape=(1, 2))
        out_identity = self.model.call(x_input, isidentity=True)
        assert tf.shape(out_identity).numpy().tolist() == [1, 2], "matrix is incorrect shape"
        np.allclose([[1, 2.]], out_identity)

    def test_identity_model_when_isidentity(self):
        x_input = tf.constant([[1, 2.]], shape=(1, 2))
        out_activate = self.model.call(x_input, isidentity=False)
        assert isinstance(out_activate, tf.Tensor), "matrix must be a tf Tensor object"
        assert tf.rank(out_activate).numpy() == 2, "matrix must be of rank 1"
        assert tf.shape(out_activate).numpy().tolist() == [1, 3], "matrix is incorrect shape"
        np.allclose([[0.26889187, 0.46381024, 0.6744604]], out_activate.numpy())


if __name__ == '__main__':
    unittest.main()
