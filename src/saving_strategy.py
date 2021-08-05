"""
Module for results' SavingStrategy interface
"""
from abc import ABC
from abc import abstractmethod


class SavingStrategy(ABC):
    """
    Results saving strategy informal interface
    """
    @abstractmethod
    def save_results(self, unscaled_test_set_x, test_set_y, prediction, save_dir):
        """
        Saves results to csv file
        """

    @abstractmethod
    def print_result(self, unscaled_test_set_x, test_set_y, prediction):
        """
        Prints out results to console
        """
