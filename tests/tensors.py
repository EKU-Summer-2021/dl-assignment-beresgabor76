import unittest
import tensorflow as tf
from src.tensors import tensor_2d
from src.tensors import tensor_4d
from src.tensors import func


class TensorsTest(unittest.TestCase):
    def test_tensor_2d(self):
        matrix = tensor_2d()
        assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
        assert tf.rank(matrix).numpy() == 2

    def test_tensor_4d(self):
        images = tensor_4d()
        assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
        assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
        assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"

    def test_func(self):
        a, b = 1.5, 2.5
        e_out = func(a, b)
        assert isinstance(e_out, tf.Tensor), "output must be a tf Tensor object"
        assert tf.rank(e_out).numpy() == 0, "output must be of rank 0"
        assert e_out.numpy() == 6.0, "output must equal 6.0"


if __name__ == '__main__':
    unittest.main()
