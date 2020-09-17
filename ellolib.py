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
        self.__x_training, self.__y_training, self.__x_test, self.__y_test = self.generate_train_test_dataset(split)
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

    def fit2(self,epoch):
        cont = 0
        number_of_rows = self.__x_training.shape[0]
        number_of_weights_adjust = 0

        for cont in range(epoch):
            print("------ Epoch {} ------".format(cont+1))
            for index in range(number_of_rows):
                x_enter = np.insert(self.__x_training[index], 0, self.__bias)
                u = (x_enter * self.__weights).sum()

                result = self.activation_function(u)
                expected_result = self.__y_training[index]

                if(result != expected_result):
                    number_of_weights_adjust += 1
                    self.__weights = self.adjust_weights(
                        x_enter, result, expected_result)
                    print("### Weights {}".format(self.__weights))
            print("### Weights Adjust {}".format(number_of_weights_adjust))
            
        self.__number_of_weights_adjust = number_of_weights_adjust
        self.__number_of_epochs = epoch
        self.__fit = True

    def predict(self):
        number_of_rows = self.__x_test.shape[0]
        results = np.array([], np.int)
        for index in range(number_of_rows):
            x_enter = np.insert(self.__x_test[index], 0, self.__bias)
            u = (x_enter * self.__weights).sum()
            results = np.append(results, self.activation_function(u)) 

        return results

    def get_confusion_matrix(self, reais, preditos, labels):
        # não implementado
        if len(labels) > 2:
            return None

        if len(reais) != len(preditos):
            return None
        
        # considerando a primeira classe como a positiva, e a segunda a negativa
        true_class = labels[0]
        negative_class = labels[1]

        # valores preditos corretamente
        tp = 0
        tn = 0
        
        # valores preditos incorretamente
        fp = 0
        fn = 0
        
        for (indice, v_real) in enumerate(reais):
            v_predito = preditos[indice]

            # se trata de um valor real da classe positiva
            if v_real == true_class:
                tp += 1 if v_predito == v_real else 0
                fp += 1 if v_predito != v_real else 0
            else:
                tn += 1 if v_predito == v_real else 0
                fn += 1 if v_predito != v_real else 0
        
        return np.array([
            # valores da classe positiva
            [ tp, fp ],
            # valores da classe negativa
            [ fn, tn ]
        ])

    def get_accuracy(self, confusion_matrix):
        tp = confusion_matrix[0][0]
        tn = confusion_matrix[1][1]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]

        return ((tp + tn) / (tp+ fp + tn + fn))
    
    def get_precision(self, confusion_matrix):
        tp = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]

        return (tp / (tp + fp))
    
    def get_recall(self, confusion_matrix):
        tp = confusion_matrix[0][0]
        fn = confusion_matrix[1][0]

        return (tp / (tp + fn))

    def get_f_score(self, precision, recall):
        return (2 * (precision * recall) / 
                    (precision + recall))

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
        
        slope = -(self.__weights[0] / self.__weights[2]) / (self.__weights[0] / self.__weights[1])
        intercept = -(self.__weights[0] / self.__weights[2])
        
        x_min = np.amin(self.__x_training[:,:1])
        x_max = np.amax(self.__x_training[:,:1])

        x = np.linspace(x_min,x_max)
        #y =mx+c, m is slope and c is inte
        y = [(slope * i) - intercept for i in x]
        return (x, y)
        
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
    file = np.fromfile("./rna-2020.1-pp2-data/data2.txt")
    file = file.reshape((int(file.shape[0] / 3), 3))
    conjunto_treinamento_aula = np.array([[2, 2, 1], [4, 4, 0]])
    b = Perceptron(dataset=file,split="holdout")
    #b.fit()
    b.fit2(100)
    confusion_matrix = b.get_confusion_matrix(b.y_test, b.predict(), [0, 1])
    accuracy = b.get_accuracy(confusion_matrix)
    precision = b.get_precision(confusion_matrix)
    recall = b.get_recall(confusion_matrix)
    f_score = b.get_f_score(precision, recall)
    print(accuracy, precision, recall, f_score)
    # print("## reta")
    # print(b.x_training.shape[0])

def questao_2():
    file = np.fromfile("./rna-2020.1-pp2-data/data2.txt")
    file = file.reshape((int(file.shape[0] / 3), 3))
    print(file)
    learn_taxs = [0.4, 0.1, 0.01]
    weights = [[-100,  100], [-1,  1], [-0.5,  0.5]]

    results = {}

    for tax in learn_taxs:
        for weight in weights:
            key = str(tax) + str(weight)
            results[key] = []
            print(key)

            for i in range(0, 101):
                b = Perceptron(
                    dataset=file, weight_random_seed=weight, learn_tax=tax)
                b.fit()

                results[key].append({
                    'hyperplane': b.generate_hyperplane(),
                    'number_of_weights_adjust': b.number_of_weights_adjust,
                    'number_of_epochs': b.number_of_epochs
                })
    print(results)