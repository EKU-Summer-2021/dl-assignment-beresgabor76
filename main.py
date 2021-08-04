from src import DatasetStudents

if __name__ == '__main__':
   dataset = DatasetStudents()
   dataset.prepare()
   print(dataset.train_set_x)
   print(dataset.train_set_y)
   print(dataset.test_set_x)
   print(dataset.test_set_y)
