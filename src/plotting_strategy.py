"""
Module for results' PlottingStrategy interface
"""
from abc import ABC
from abc import abstractmethod
import os
import pandas as pd
import matplotlib.pyplot as plt


class PlottingStrategy(ABC):
    """
    Results plotting strategy abstract class
    """
    def plot_learning_curve(self, history, save_dir):
        """
        Saves plot of DNN's learning curve to png file
        """
        pd.DataFrame(history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plot_file = os.path.join(os.path.dirname(__file__), save_dir, 'learning_curve.png')
        plt.savefig(plot_file)

    @abstractmethod
    def plot_results(self, test_set_y, prediction, save_dir):
        """
        Saves plot results of DNN's test to png file
        """

    @abstractmethod
    def plot_show(self, test_set_y, prediction):
        """
        Shows plot results of DNN's test
        """
