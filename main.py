import tensorflow as tf
from src.tensors import tensor_2d
from src.tensors import tensor_4d
from src.tensors import func
from src import OurDenseLayer
from src import SequentialModel
from src import SubclassModel
from src import IdentityModel
from src.function_minimization import function_minimization


if __name__ == '__main__':
    #Task 1.1
    tensor_2d()
    tensor_4d()
    #Task 1.2
    a, b = 1.5, 2.5
    e_out = func(a, b)
    print(e_out)
    #Task 1.3.1
    tf.random.set_seed(1)
    layer = OurDenseLayer(3)
    layer.build((1, 2))
    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    y = layer.call(x_input)
    # test the output
    print(y.numpy())
    #Task 1.3.2
    tf.random.set_seed(1)
    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    model = SequentialModel(n_output_nodes=3, input_shape=x_input.shape)
    model_output = model.call(x_input)
    print(model_output)
    # Task 1.3.3
    tf.random.set_seed(1)
    n_output_nodes = 3
    model = SubclassModel(n_output_nodes)
    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    print(model.call(x_input))
    # Task 1.3.4
    tf.random.set_seed(1)
    n_output_nodes = 3
    model = IdentityModel(n_output_nodes)
    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    '''TODO: pass the input into the model and call with and without the input identity option.'''
    out_activate = model.call(x_input, isidentity=False)
    out_identity = model.call(x_input, isidentity=True)
    print("Network output with activation: {}; network identity output: {}"
          .format(out_activate.numpy(), out_identity.numpy()))
    #Task 1.4
    function_minimization()


