"""
Module for Dataset class for Students grade prediction
"""
import pandas as pd

from src import Dataset


class DatasetStudents(Dataset):
    """
    Dataset class for Students grade prediction
    """
    def __init__(self):
        super().__init__(filenames=['student-mat.csv', 'student-por.csv'])

    def _feature_selection(self):
        """
        Not relevant attributes are dropped
        """
        self._dataset.drop('G1', axis=1, inplace=True)
        self._dataset.drop('G2', axis=1, inplace=True)
        corr_matrix = self._dataset.corr()
        columns = corr_matrix['G3']
        columns = columns[abs(columns.values) < 0.1]
        for column_name in columns.index:
            try:
                self._dataset.drop(column_name, axis=1, inplace=True)
            except:
                pass

    def _categories_encoding(self):
        """
        Categories converted to numerical attributes, y attribute preserved as last column
        """
        dataset_y = self._dataset.iloc[:, -1]
        self._dataset.drop('G3', axis=1, inplace=True)
        self._category_ordinal_encoder('school')
        self._category_ordinal_encoder('sex')
        self._category_ordinal_encoder('address')
        self._category_ordinal_encoder('famsize')
        self._category_ordinal_encoder('Pstatus')
        self._category_1hot_encoder('Mjob')
        self._category_1hot_encoder('Fjob')
        self._category_1hot_encoder('reason')
        self._category_1hot_encoder('guardian')
        self._category_ordinal_encoder('schoolsup')
        self._category_ordinal_encoder('famsup')
        self._category_ordinal_encoder('paid')
        self._category_ordinal_encoder('activities')
        self._category_ordinal_encoder('nursery')
        self._category_ordinal_encoder('higher')
        self._category_ordinal_encoder('internet')
        self._category_ordinal_encoder('romantic')
        self._dataset = pd.concat([self._dataset, dataset_y], axis=1)

    def _split_dataset_into_x_y(self):
        """
        Splits dataset  into x and y sets according to the feature arrangement
        """
        self.train_set_y = self.train_set_x[self.train_set_x.columns[-1]].copy()
        self.train_set_x = self.train_set_x.drop(self.train_set_x.columns[-1], axis=1).copy()
        self.test_set_y = self.test_set_x[self.test_set_x.columns[-1]].copy()
        self.test_set_x = self.test_set_x.drop(self.test_set_x.columns[-1], axis=1).copy()
