"""
Defining a model using subclassing
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class SubclassModel(Model):
    # In __init__, we define the Model's layers
    def __init__(self, n_output_nodes):
        super(SubclassModel, self).__init__()
        '''TODO: Our model consists of a single Dense layer. Define this layer.'''
        self.dense_layer = Dense(n_output_nodes,
                                 activation='sigmoid',
                                 use_bias=True,
                                 kernel_initializer=None,  # 'random ?
                                 bias_initializer=None,  # 'random ?
                                 kernel_regularizer=None,
                                 bias_regularizer=None,
                                 activity_regularizer=None,
                                 kernel_constraint=None,
                                 bias_constraint=None)

    # In the call function, we define the Model's forward pass.
    def call(self, inputs):
        return self.dense_layer(inputs)
