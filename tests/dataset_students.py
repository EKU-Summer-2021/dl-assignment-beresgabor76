import unittest
from src import DatasetStudents


class DatasetStudentsTest(unittest.TestCase):
    def test_prepare(self):
        data = DatasetStudents()
        data.prepare()
        total_rows = data._dataset.shape[0]
        self.assertEqual(round(total_rows * 0.8), data.train_set_x.shape[0])
        self.assertEqual(round(total_rows * 0.2), data.test_set_x.shape[0])


if __name__ == '__main__':
    unittest.main()
