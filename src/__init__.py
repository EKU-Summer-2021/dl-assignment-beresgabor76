'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from src.polynomial import Polynomial
from src.dataset import Dataset
from src.dataset_students import DatasetStudents

__all__ = [
    'Polynomial',
    'Dataset',
    'DatasetStudents'
]
