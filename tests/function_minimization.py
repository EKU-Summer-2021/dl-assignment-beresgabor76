import unittest
import os
from src.function_minimization import function_minimization


class FunctionMinimizationTest(unittest.TestCase):
    def test_function_minimization(self):
        function_minimization()
        filepath = os.path.join(os.path.dirname(__file__), '../results/tutorial', 'minimization.png')
        self.assertEqual(True, os.path.exists(filepath))  # add assertion here


if __name__ == '__main__':
    unittest.main()
