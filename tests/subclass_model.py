import unittest
import numpy as np
import tensorflow as tf
from src import SubclassModel


class SubclassModelTest(unittest.TestCase):

    def test_subclass_model(self):
        tf.random.set_seed(1)
        n_output_nodes = 3
        model = SubclassModel(n_output_nodes)
        x_input = tf.constant([[1, 2.]], shape=(1, 2))
        model_output = model.call(x_input)
        assert isinstance(model_output, tf.Tensor), "matrix must be a tf Tensor object"
        assert tf.rank(model_output).numpy() == 2, "matrix must be of rank 1"
        assert tf.shape(model_output).numpy().tolist() == [1, 3], "matrix is incorrect shape"
        np.allclose([[0.26889187, 0.46381024, 0.6744604]], model_output.numpy())


if __name__ == '__main__':
    unittest.main()
