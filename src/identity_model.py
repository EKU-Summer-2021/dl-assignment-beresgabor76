"""
Defining a model using subclassing and specifying custom behavior ###
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class IdentityModel(Model):
    # As before, in __init__ we define the Model's layers
    # Since our desired behavior involves the forward pass, this part is unchanged
    def __init__(self, n_output_nodes):
        super(IdentityModel, self).__init__()
        self.dense_layer = Dense(n_output_nodes, activation='sigmoid')

    '''TODO: Implement the behavior where the network outputs the input, unchanged, 
        under control of the isidentity argument.'''
    def call(self, inputs, isidentity=False):
        x = self.dense_layer(inputs)
        '''TODO: Implement identity behavior'''
        if isidentity:
            return inputs
        else:
            return x
