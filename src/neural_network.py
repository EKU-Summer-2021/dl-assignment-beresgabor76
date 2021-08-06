"""
Module for neural network class implementation
"""
import sys
import os
import datetime
import logging
import numpy as np
import pandas as pd
import tensorflow as tf


class NeuralNetwork:
    """
    Class for neural network class implementation
    """

    def __init__(self, input_number, layer_sizes, activation_functions, saving_strategy, plotting_strategy):
        self.__parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '../results/students_grade')
        self.__sub_dir = self.__make_save_dir()
        self.__logger = self.__setup_logger(f'DnnLog{self.__sub_dir}',
                                            os.path.join(self.__parent_dir, self.__sub_dir, 'run.log'))
        self.__dnn = None
        self.__build_dnn(input_number, layer_sizes, activation_functions)
        self.__history = None

        self.__plotting_strategy = plotting_strategy
        self.__saving_strategy = saving_strategy
        self.__test_set_x = None
        self.__test_set_y = None
        self.__prediction = None

    def __build_dnn(self, input_number, layer_sizes, activation_functions):
        if input_number < 1:
            print('Input number must be greater than zero!')
            sys.exit(1)
        if len(layer_sizes) < 2:
            print("At least 2 layer sizes must be given as the DNN's hidden and output layer!")
            sys.exit(1)
        if len(activation_functions) < 2:
            print("At least 2 activation function must be given for the DNN's hidden and output layers!")
            sys.exit(1)
        self.__dnn = tf.keras.Sequential()
        self.__dnn.add(tf.keras.layers.InputLayer(input_shape=(input_number,)))
        for i in range(len(layer_sizes) - 1):
            if i < len(activation_functions) - 1:
                self.__dnn.add(tf.keras.layers.Dense(units=layer_sizes[i],
                                                     activation=activation_functions[i],
                                                     kernel_initializer=None,
                                                     bias_initializer='zeros'))
            else:
                self.__dnn.add(tf.keras.layers.Dense(units=layer_sizes[i],
                                                     activation=activation_functions[len(activation_functions) - 2],
                                                     kernel_initializer=None,
                                                     bias_initializer='zeros'))

        self.__dnn.add(tf.keras.layers.Dense(units=layer_sizes[len(layer_sizes) - 1],
                                             activation=activation_functions[len(activation_functions) - 1],
                                             kernel_initializer=None,
                                             bias_initializer='zeros'))
        self.__dnn.compile(loss='sparse_categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])
        self.__dnn.summary(print_fn=lambda x: self.__logger.info(x))
        self.__logger.info('activation functions: %s', str(activation_functions))

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

    def __setup_logger(self, name, log_file, level=logging.INFO):
        """
        Setup loggers for the learning algorithms
        """
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    def train(self, train_set_x, train_set_y, early_stopping=True, epochs=500):
        """
        Trains DNN with given dataset
        """
        if early_stopping:
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True)
            self.__dnn.fit(x=train_set_x, y=train_set_y, epochs=epochs,
                           validation_split=0.2, callbacks=[early_stopping_cb])
        else:
            self.__dnn.fit(x=train_set_x, y=train_set_y, epochs=epochs,
                           validation_split=0.2)
        self.__history = self.__dnn.history.history
        self.__logger.info(self.__history)

    def test(self, test_set_x, test_set_y):
        """
        Tests the trained neural network on given test set
        """
        self.__test_set_x = test_set_x
        self.__test_set_y = test_set_y
        results = self.__dnn.evaluate(x=test_set_x, y=test_set_y)
        self.__logger.info('test loss, test accuracy: %s', str(np.round(results, 2)))
        proba_mx = self.__dnn.predict(x=test_set_x)
        classes = []
        for probas in proba_mx:
            classes.append(np.argmax(probas))
        self.__prediction = pd.Series(classes, name='Prediction')

    def plot_test_results(self):
        """
        Plots out test results as histogram and confusion matrix to png files
        """
        self.__plotting_strategy.plot_results(self.__test_set_y,
                                              self.__prediction,
                                              self.__parent_dir + '/' + self.__sub_dir)

    def plot_learning_curve(self):
        """
        Plots out training learning curve to png files
        """
        self.__plotting_strategy.plot_learning_curve(self.__history,
                                                     self.__parent_dir + '/' + self.__sub_dir)

    def save_results(self, x_scaler=None, y_scaler=None):
        """
        Saves test dataset with target values and prediction results with errors
        """
        if x_scaler is not None:
            unscaled_test_set_x = pd.DataFrame(x_scaler.inverse_transform(self.__test_set_x),
                                               columns=self.__test_set_x.columns)
        else:
            unscaled_test_set_x = self.__test_set_x
        if y_scaler is not None:
            unscaled_test_set_y = pd.DataFrame(y_scaler.inverse_transform(self.__test_set_y),
                                               columns=self.__test_set_y.columns)
            unscaled_prediction = pd.DataFrame(y_scaler.inverse_transform(self.__prediction),
                                               columns=self.__prediction.columns)
        else:
            unscaled_test_set_y = self.__test_set_y
            unscaled_prediction = self.__prediction
        self.__saving_strategy.save_results(unscaled_test_set_x,
                                            unscaled_test_set_y, unscaled_prediction,
                                            self.__parent_dir + '/' + self.__sub_dir)
