'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from src.polynomial import Polynomial
from src.dataset import Dataset
from src.dataset_students import DatasetStudents
from src.plotting_strategy import PlottingStrategy
from src.plotting_strategy_clf import PlottingStrategy4CLF
from src.saving_strategy import SavingStrategy
from src.saving_strategy_sl import SavingStrategy4SL
from src.neural_network import NeuralNetwork


__all__ = [
    'Polynomial',
    'Dataset',
    'DatasetStudents',
    'PlottingStrategy4CLF',
    'SavingStrategy4SL',
    'NeuralNetwork'
]
