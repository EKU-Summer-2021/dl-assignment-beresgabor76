from src import DatasetStudents
from src import NeuralNetwork
from src import PlottingStrategy4CLF
from src import SavingStrategy4SL

if __name__ == '__main__':
   dataset = DatasetStudents()
   dataset.prepare()
   dnn = NeuralNetwork(input_number=dataset.train_set_x.shape[1],
                       layer_sizes=[300, 100, 50, 21],
                       activation_functions=['relu', 'softmax'],
                       saving_strategy=SavingStrategy4SL(),
                       plotting_strategy=PlottingStrategy4CLF())
   #dnn.train(dataset.train_set_x, dataset.train_set_y, early_stopping=True, epochs=500)
   #dnn.plot_learning_curve()
   dnn.train(dataset.train_set_x, dataset.train_set_y, early_stopping=False, epochs=250)
   dnn.plot_learning_curve()
   dnn.test(dataset.test_set_x, dataset.test_set_y)
   dnn.plot_test_results()
