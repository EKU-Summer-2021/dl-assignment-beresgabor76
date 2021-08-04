"""
Module for basic tensorflow operations
"""
import tensorflow as tf
import numpy as np


def tensor_2d():
    """Define a 2-d Tensor"""
    matrix = tf.random.uniform((4, 5), 0, 10, tf.int32)
    assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
    assert tf.rank(matrix).numpy() == 2
    return matrix


def tensor_4d():
    """Define a 4-d Tensor."""
    # Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.
    #   You can think of this as 10 images where each image is RGB 256 x 256.
    images = tf.zeros(shape=(10, 256, 256, 3))
    assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
    assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
    assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
    return images


def func(a, b):
    """Define the operation for c, d, e (use tf.add, tf.subtract, tf.multiply)."""
    c = tf.add(a, b)
    d = tf.subtract(b, 1)
    e = tf.multiply(c, d)
    return e
