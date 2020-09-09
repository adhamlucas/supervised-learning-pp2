import numpy as np

#numpy = np.loadtxt('./rna-2020.1-pp2-data/dataAll.txt')

class perceptron:
  def __init__(self, dataset, weight_random_seed=[-0.5, 0.5], bias=-1, degree=0, learn_tax=0.1):
    self.__dataset = dataset
    self.__learn_tax = learn_tax
    self.__degree = degree
    self.__x_training, self.__y_training, self.__x_test, self.__y_test = self.generate_train_test_dataset()
    self.__weights = np.random.uniform(low=weight_random_seed[0],high=weight_random_seed[1], size=3)
  
  def fit():
    error = True
    while(error):
      print('i')

 

  def activation_function(self, value):
    return (1 if value >= self.__degree else 0)

  def generate_weights(self):
    return 0

  def generate_train_test_dataset(self):
    ## verificar se a geração de números randomicos não se repete para os dataset de treino e teste de forma que os datasets gerados estejam com valores repetidos ou nem usem o tamanho total do dataset
    dataset_len_row = self.__dataset.shape[0]
    tirthy_percenty = round(dataset_len_row * 0.3)
    seventy_percenty = round(dataset_len_row * 0.7)

    training_idx = np.random.randint(self.__dataset.shape[0], size=seventy_percenty)
    test_idx = np.random.randint(self.__dataset.shape[0], size=tirthy_percenty)

    training, test = self.__dataset[training_idx,:], self.__dataset[test_idx,:]
    x_training = np.array((training[0:,:2]))
    y_training = np.array((training[0:,2:]))
    x_test = np.array((test[0:,:2]))
    y_test = np.array((test[0:,2:]))

    return (x_training, y_training, x_test, y_test)

  @property
  def degree(self):
    return self.__degree

  @property
  def x_training(self):
    return self.__x_training
  
  @property
  def y_training(self):
    return self.__y_training

  @property
  def x_test(self):
    return self.__x_test
  
  @property
  def y_test(self):
    return self.__y_test

  @property
  def weights(self):
    return self.__weights

if __name__ == "__main__":
  file = np.fromfile("./rna-2020.1-pp2-data/dataAll.txt")
  file = file.reshape((int(file.shape[0] / 3), 3))
  a = perceptron(dataset=file)
  #print(file)
  print((a.x_training[0] * a.weights[1:]).sum())
  print(a.x_training[0], a.weights[1:])