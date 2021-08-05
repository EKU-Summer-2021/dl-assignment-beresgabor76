"""
Module for dataset class preparing for processing in neural network
"""
import sys
import os
from abc import ABC
from abc import abstractmethod
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Dataset(ABC):
    """
    Abstract class implementing template pattern for storing and preparing data
    for deep learning
    """

    def __init__(self, filenames, test_size=0.2, random_state=20):
        super().__init__()
        self._filenames = filenames
        self._test_size = test_size
        self._random_state = random_state
        self._dataset = None
        self.train_set_x = None
        self.train_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        self.x_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        self.y_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)

    def prepare(self):
        """
        Prepares dataset for machine learning process
        """
        self.__read_file()
        self._feature_selection()
        self._categories_encoding()
        self.__split_dataset_into_train_test()
        self._split_dataset_into_x_y()
        self.__feature_scaling()

    def __read_file(self):
        """
        Reads in a csv file and splits into X and y sets
        """
        try:
            for filename in self._filenames:
                dataset = pd.read_csv(
                    os.path.join(os.path.dirname(__file__), '../data', filename))
                self._dataset = pd.concat([self._dataset, dataset], axis=0, ignore_index=True)
        except:
            print('An error occurred during reading the csv file.')
            sys.exit(1)

    @abstractmethod
    def _feature_selection(self):
        """
        Used in child classes to drop not relevant attributes
        """

    @abstractmethod
    def _categories_encoding(self):
        """
        Used in child classes to encode category attributes
        """

    def _category_ordinal_encoder(self, column_name, categories='auto'):
        """
        Encodes a category attribute to ordinal numbers
        """
        dataset_cat = self._dataset[[column_name]]
        cat_encoder = OrdinalEncoder(categories=categories)
        arr_cat_ordinal = cat_encoder.fit_transform(dataset_cat)
        df_cat_ordinal = pd.DataFrame(arr_cat_ordinal, columns=[column_name])
        self._dataset.drop(column_name, axis=1, inplace=True)
        self._dataset = pd.concat([self._dataset, df_cat_ordinal], axis=1)

    def _category_1hot_encoder(self, column_name):
        """
        Encodes a category attribute to a set of attributes, one for each category value
        """
        dataset_cat = self._dataset[[column_name]]
        cat_encoder = OneHotEncoder()
        arr_cat_1hot = cat_encoder.fit_transform(dataset_cat)
        df_cat_1hot = pd.DataFrame(arr_cat_1hot.toarray(), columns=cat_encoder.get_feature_names())
        self._dataset = pd.concat([self._dataset, df_cat_1hot], axis=1) \
            .drop(column_name, axis=1)

    def __split_dataset_into_train_test(self):
        """
        Splits dataset to train and test sets
        """
        self.train_set_x, self.test_set_x = train_test_split(self._dataset,
                                                             test_size=self._test_size,
                                                             random_state=self._random_state)
        self.train_set_x.reset_index(inplace=True)
        self.test_set_x.reset_index(inplace=True)

    @abstractmethod
    def _split_dataset_into_x_y(self):
        """
        Implemented in child classes according to x and y feature numbers
        """

    def __feature_scaling(self):
        """
        Scales down all input data to [0, 1] interval
        """
        self.x_scaler.fit(self.train_set_x)
        scaled_arr = self.x_scaler.transform(self.train_set_x)
        self.train_set_x = pd.DataFrame(scaled_arr, columns=self.train_set_x.columns)
        # self.y_scaler.fit(np.array([self.train_set_y]).reshape(-1, 1))
        # scaled_arr = self.y_scaler.transform(np.array([self.train_set_y]).reshape(-1, 1))
        # self.train_set_y = np.ravel(scaled_arr)
        scaled_arr = self.x_scaler.transform(self.test_set_x)
        self.test_set_x = pd.DataFrame(scaled_arr, columns=self.test_set_x.columns)
        # scaled_arr = self.y_scaler.transform(np.array([self.test_set_y]).reshape(-1, 1))
        # self.test_set_y = pd.DataFrame(scaled_arr)
