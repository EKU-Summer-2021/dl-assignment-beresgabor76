import unittest
import os
from src import DatasetStudents
from src import NeuralNetwork
from src import SavingStrategy4SL
from src import PlottingStrategy4CLF


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        self.data = DatasetStudents()
        self.data.prepare()
        self.dnn = NeuralNetwork(input_number=self.data.train_set_x.shape[1],
                                 layer_sizes=[100, 50, 21],
                                 activation_functions=['relu', 'softmax'],
                                 saving_strategy=SavingStrategy4SL(),
                                 plotting_strategy=PlottingStrategy4CLF())

    def test_initialize(self):
        self.assertEqual(True, self.dnn._NeuralNetwork__dnn.weights[0].numpy().all())
        self.assertEqual(False, self.dnn._NeuralNetwork__dnn.weights[1].numpy().any())
        self.assertEqual(True, self.dnn._NeuralNetwork__dnn.weights[2].numpy().all())
        self.assertEqual(False, self.dnn._NeuralNetwork__dnn.weights[3].numpy().any())
        self.assertEqual(True, self.dnn._NeuralNetwork__dnn.weights[4].numpy().all())
        self.assertEqual(False, self.dnn._NeuralNetwork__dnn.weights[5].numpy().any())

    def test_train(self):
        self.dnn.train(self.data.train_set_x, self.data.train_set_y,
                       early_stopping=True, epochs=250)
        self.assertEqual(True, self.dnn._NeuralNetwork__dnn.evaluate(
            self.data.test_set_x, self.data.test_set_y)[1] > 0.1)

    def test_test(self):
        self.dnn.train(self.data.train_set_x, self.data.train_set_y,
                       early_stopping=True, epochs=250)
        self.dnn.test(self.data.test_set_x, self.data.test_set_y)
        total_rows = self.data._dataset.shape[0]
        self.assertEqual((round(total_rows * 0.2),), self.dnn._NeuralNetwork__prediction.shape)

    def test_plot_results(self):
        self.dnn.train(self.data.train_set_x, self.data.train_set_y,
                       early_stopping=True, epochs=250)
        self.dnn.test(self.data.test_set_x, self.data.test_set_y)
        self.dnn.plot_test_results()
        plot_file = os.path.join(os.path.dirname(__file__),
                                 self.dnn._NeuralNetwork__parent_dir + '/'
                                 + self.dnn._NeuralNetwork__sub_dir,
                                 'histogram.png')
        self.assertEqual(True, os.path.isfile(plot_file))
        plot_file = os.path.join(os.path.dirname(__file__),
                                 self.dnn._NeuralNetwork__parent_dir + '/'
                                 + self.dnn._NeuralNetwork__sub_dir,
                                 'confusion_mx.png')
        self.assertEqual(True, os.path.isfile(plot_file))

    def test_plot_learning_curve(self):
        self.dnn.train(self.data.train_set_x, self.data.train_set_y,
                       early_stopping=True, epochs=250)
        self.dnn.test(self.data.test_set_x, self.data.test_set_y)
        self.dnn.plot_learning_curve()
        plot_file = os.path.join(os.path.dirname(__file__),
                                 self.dnn._NeuralNetwork__parent_dir + '/'
                                 + self.dnn._NeuralNetwork__sub_dir,
                                 'learning_curve.png')
        self.assertEqual(True, os.path.isfile(plot_file))

    def test_save_results(self):
        self.dnn.train(self.data.train_set_x, self.data.train_set_y,
                       early_stopping=True, epochs=250)
        self.dnn.test(self.data.test_set_x, self.data.test_set_y)
        self.dnn.save_results()
        csv_file = os.path.join(os.path.dirname(__file__),
                                self.dnn._NeuralNetwork__parent_dir + '/'
                                + self.dnn._NeuralNetwork__sub_dir,
                                'results.csv')
        self.assertEqual(True, os.path.isfile(csv_file))


if __name__ == '__main__':
    unittest.main()
