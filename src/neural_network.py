"""
Module for neural network class implementation
"""
import os
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


class NeuralNetwork:
    """
    Class for neural network class implementation
    """

    def __init__(self, input_number, layer_sizes, activation_functions, saving_strategy, plotting_strategy):
        self.__dnn = None
        self.__build_dnn(input_number, layer_sizes, activation_functions)
        self.__history = None
        self.__parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '../results/students_grade')
        self.__sub_dir = self.__make_save_dir()
        self.__plotting_strategy = plotting_strategy
        self.__saving_strategy = saving_strategy
        self.__test_set_y = None
        self.__prediction = None

    def __build_dnn(self, input_number, layer_sizes, activation_functions):
        if input_number < 1:
            print('Input number must be greater than zero!')
            exit(1)
        if len(layer_sizes) < 2:
            print("At least 2 layer sizes must be given as the DNN's hidden and output layer!")
            exit(1)
        if len(activation_functions) < 2:
            print("At least 2 activation function must be given for the DNN's hidden and output layers!")
            exit(1)
        self.__dnn = Sequential()
        self.__dnn.add(InputLayer(input_shape=(input_number,)))
        for i in range(len(layer_sizes) - 1):
            if i < len(activation_functions) - 1:
                self.__dnn.add(Dense(units=layer_sizes[i],
                                     activation=activation_functions[i],
                                     kernel_initializer=None,
                                     bias_initializer=None))
            else:
                self.__dnn.add(Dense(units=layer_sizes[i],
                                     activation=activation_functions[len(activation_functions) - 2],
                                     kernel_initializer=None,
                                     bias_initializer=None))

        self.__dnn.add(Dense(units=layer_sizes[len(layer_sizes) - 1],
                             activation=activation_functions[len(activation_functions) - 1],
                             kernel_initializer=None,
                             bias_initializer=None))
        self.__dnn.compile(loss='sparse_categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def __make_save_dir(self):
        """
        Makes a subdirectory for storing result and plot
        """
        resolution = datetime.timedelta(seconds=5)
        save_time = datetime.datetime.now() - datetime.timedelta(
            seconds=datetime.datetime.now().second % resolution.seconds)
        sub_dir = save_time.strftime('%Y.%m.%d_%H.%M.%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), self.__parent_dir, sub_dir)):
            os.chdir(self.__parent_dir)
            os.mkdir(sub_dir)
        return sub_dir

    def train(self, train_set_x, train_set_y, early_stopping=True, epochs=500):
        """
        Trains DNN with given dataset
        """
        if early_stopping:
            early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
            self.__dnn.fit(x=train_set_x, y=train_set_y, epochs=epochs,
                           validation_split=0.2, callbacks=[early_stopping_cb])
        else:
            self.__dnn.fit(x=train_set_x, y=train_set_y, epochs=epochs,
                           validation_split=0.2)
        self.__history = self.__dnn.history.history

    def test(self, test_set_x, test_set_y):
        self.__test_set_y = test_set_y
        results = self.__dnn.evaluate(x=test_set_x, y=test_set_y)
        print('test loss, test accuracy:', np.round(results, 2))
        self.__prediction = pd.Series(self.__dnn.predict_classes(x=test_set_x), name='Prediction')

    def plot_test_results(self):
        self.__plotting_strategy.plot_results(self.__test_set_y,
                                              self.__prediction,
                                              self.__parent_dir + '/' + self.__sub_dir)

    def plot_learning_curve(self):
        self.__plotting_strategy.plot_history(self.__history,
                                              self.__parent_dir + '/' + self.__sub_dir)