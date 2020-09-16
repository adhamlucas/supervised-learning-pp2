import numpy as np


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class NoFitError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Perceptron:
    def __init__(self, dataset, weight_random_seed=[-0.5, 0.5], bias=-1, degree=0, learn_tax=0.1, split="without"):
        self.__dataset = dataset
        self.__learn_tax = learn_tax
        self.__degree = degree
        self.__bias = bias
        self.__x_training, self.__y_training, self.__x_test, self.__y_test = self.generate_train_test_dataset(
            split)
        self.__weights = self.generate_weights(weight_random_seed)
        self.__fit = False
        self.__number_of_weights_adjust = 0
        self.__number_of_epochs = 0

    def fit(self):
        epoch = 1
        error = True
        number_of_rows = self.__x_training.shape[0]
        number_of_weights_adjust = 0
        while(error):
            print("------ Epoch {} ------".format(epoch))
            epoch += 1
            erros_count = 0

            for index in range(number_of_rows):
                x_enter = np.insert(self.__x_training[index], 0, self.__bias)
                u = (x_enter * self.__weights).sum()

                result = self.activation_function(u)
                expected_result = self.__y_training[index]

                if(result != expected_result):
                    erros_count += 1
                    number_of_weights_adjust += 1
                    self.__weights = self.adjust_weights(
                        x_enter, result, expected_result)
                    print("### Weights {}".format(self.__weights))

            print("### Weights Adjust {}".format(number_of_weights_adjust))

            if(erros_count == 0):
                self.__number_of_weights_adjust = number_of_weights_adjust
                self.__number_of_epochs = epoch
                error = False

        print('{0} \nTotal Weights\' adjust={1}'.format(
            '-' * 30, number_of_weights_adjust))
        self.__fit = True

    def activation_function(self, value):
        return (1 if value >= self.__degree else 0)

    def adjust_weights(self, x_enter, result, expected_result):
        new_weights = self.__weights + \
            (self.__learn_tax * (expected_result - result) * x_enter)
        return new_weights

    def generate_weights(self, weight_random_seed):
        return np.random.uniform(low=weight_random_seed[0], high=weight_random_seed[1], size=3)

    def generate_train_test_dataset(self, split):
        # verificar se a geração de números randomicos não se repete para os dataset de treino e teste de forma que os datasets gerados estejam com valores repetidos ou nem usem o tamanho total do dataset
        dataset_len_row = self.__dataset.shape[0]
        # condição a fim de testes com a entada pequena da aula
        if(split == "without"):
            x_training = np.array((self.__dataset[0:, :2]))
            y_training = np.array((self.__dataset[0:, 2:]))
            return (x_training, y_training, np.array((0)), np.array((0)))

        tirthy_percenty = round(dataset_len_row * 0.3)
        seventy_percenty = round(dataset_len_row * 0.7)

        training_idx = np.random.randint(
            dataset_len_row, size=seventy_percenty)
        test_idx = np.random.randint(dataset_len_row, size=tirthy_percenty)

        training, test = self.__dataset[training_idx,
                                        :], self.__dataset[test_idx, :]
        x_training = np.array((training[0:, :2]))
        y_training = np.array((training[0:, 2:]))
        x_test = np.array((test[0:, :2]))
        y_test = np.array((test[0:, 2:]))

        return (x_training, y_training, x_test, y_test)

    def generate_hyperplane(self):
        if(not self.__fit):
            raise NoFitError(self.__fit, "Error: Perceptron isn't trained")

        first_point_x = 0
        first_point_y = self.__weights[0] / self.__weights[2] + self.__weights[1] / self.__weights[2] * first_point_x
        first_point = (first_point_x, first_point_y)

        second_point_y = 0
        second_point_x = -(self.__weights[0] / self.__weights[2]) + self.__weights[1] / self.__weights[2]
        second_point = (second_point_x, second_point_y)

        return (first_point, second_point)

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
    def number_of_weights_adjust(self):
        return self.__number_of_weights_adjust

    @property
    def number_of_epochs(self):
        return self.__number_of_epochs


if __name__ == "__main__":
    file = np.fromfile("./rna-2020.1-pp2-data/dataAll.txt")
    file = file.reshape((int(file.shape[0] / 3), 3))
    conjunto_treinamento_aula = np.array([[2, 2, 1], [4, 4, 0]])
    b = Perceptron(dataset=conjunto_treinamento_aula)
    b.fit()
    print("## reta")
    print(b.generate_hyperplane())
    print(b.x_training.shape[0])


def questao_2():
    file = np.fromfile("./rna-2020.1-pp2-data/data2.txt")
    file = file.reshape((int(file.shape[0] / 3), 3))
    learn_taxs = [0.4, 0.1, 0.01]
    weights = [[-100,  100], [-1,  1], [-0.5,  0.5]]

    results = []

    for tax in learn_taxs:
        for weight in weights:
            for i in range(0, 101):
                b = Perceptron(
                    dataset=file, weight_random_seed=weight, learn_tax=tax)
                b.fit()

                results.append({
                    'hyperplane': b.generate_hyperplane(),
                    'number_of_weights_adjust': b.number_of_weights_adjust,
                    'number_of_epochs': b.number_of_epochs
                })
    print(results)
