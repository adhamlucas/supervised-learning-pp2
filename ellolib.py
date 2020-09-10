import numpy as np

class Perceptron:
  def __init__(self, dataset, weight_random_seed=[-0.5, 0.5], bias=-1, degree=0, learn_tax=0.1):
    self.__dataset = dataset
    self.__learn_tax = learn_tax
    self.__degree = degree
    self.__bias = bias
    self.__x_training, self.__y_training, self.__x_test, self.__y_test = self.generate_train_test_dataset()
    self.__errors = np.full(self.__y_training.shape[0], False)
    self.__weights = self.generate_weights(weight_random_seed)
  
  def fit(self):
    epoch = 0
    error = True
    while(error):
      print("### Epoch {} ###".format(epoch))
      print("### Weights {}".format(self.__weights))
      epoch += 1
      erros_count = 0

      for index in range(self.__x_training.shape[0]):
      #  print("x_traing={1} weights={2} sum={3}".format(self.__weights[0], self.__x_training[index], self.__weights, (self.__x_training[index] * self.__weights[1:]).sum()))
        u = self.__bias * self.__weights[0] + (self.__x_training[index] * self.__weights[1:]).sum()
        result = self.activation_function(u)
      #  print("### u = {}".format(u))
        expected_result = self.__y_training[index]

        if(result != expected_result):
         # print("here")
          erros_count += 1
          self.__weights = self.adjust_weights(self.__x_training[index], result, expected_result)

      print("### Erros count{}".format(erros_count))
      
      if(erros_count == 0):
        error = False

  def activation_function(self, value):
    return (1 if value >= self.__degree else -10)

  def adjust_weights(self, enter, result, expected_result):
    enter = np.insert(enter, 0, self.__bias)
    new_weights = self.__weights + (self.__learn_tax * (expected_result - result) * enter)
    return new_weights

  def generate_weights(self, weight_random_seed):
    return np.random.uniform(low=weight_random_seed[0],high=weight_random_seed[1], size=3)

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

  @property
  def errors(self):
    return self.__errors

if __name__ == "__main__":
  file = np.fromfile("./rna-2020.1-pp2-data/dataAll.txt")
  file = file.reshape((int(file.shape[0] / 3), 3))
  a = Perceptron(dataset=file)
  a.fit()
  print("--------------")
  print(np.insert(a.x_training[0], 0, -1))
  print(a.weights[0])