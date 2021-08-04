import unittest
import numpy as np
import tensorflow as tf
from src import SequentialModel


class SequentialModelTest(unittest.TestCase):

    def test_sequential_model(self):
        tf.random.set_seed(1)
        x_input = tf.constant([[1, 2.]], shape=(1, 2))
        model = SequentialModel(n_output_nodes=3, input_shape=x_input.shape)
        model_output = model.call(x_input)
        assert isinstance(model_output, tf.Tensor), "matrix must be a tf Tensor object"
        assert tf.rank(model_output).numpy() == 2, "matrix must be of rank 2"
        assert tf.shape(model_output).numpy().tolist() == [1, 3], "matrix is incorrect shape"
        np.allclose([[0.26889187, 0.46381024, 0.6744604]], model_output.numpy())


if __name__ == '__main__':
    unittest.main()
