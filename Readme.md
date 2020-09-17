# Projeto

## Sobre
O objetivo deste projeto é a implementação de um algoritmo de treinamento mediante Aprendizado Supervisionado do neurônio Perceptron de Rosenblatt aplicado em problemas de classificação.


Projeto desenvolvido na disciplina de Redes Neurais Artificiais.

Orientadora: [Prof. Dra. Elloá B. Guedes](https://github.com/elloa)

---

## Tabela de conteúdos
<!--ts-->
   * [Sobre](#sobre)
   * [Tabela de Conteudo](#tabela-de-conteúdos)
   * [Como usar](#como-usar)
      * [Pré-requisitos](#pré-requisitos)
      * [Dataset utilizado](#dataset-utilizado)
      * [Executando o projeto](#executando-o-projeto)
   * [Tecnologias](#computer-tecnologias)
   * [Ellolib](#ellolib)
   * [Autores](#autores)
<!--te-->

---

## Como usar

### Pré-requisitos
Para começar é necessário a linguagem [Python3+](https://www.python.org/), o [Git](https://git-scm.com/) e as seguintes bibliotecas:

  * [NumPy](https://numpy.org/)
  * [Matplotlib](https://matplotlib.org/)

É recomendado que você instale a distribuição [Anaconda](https://www.anaconda.com/products/individual), pois ela já possui muitos pacotes e bibliotecas.

### Dataset utilizado
As atividades tem como entrada um arquivo *txt* com conteúdo em binário descrevendo um *numpy.ndarray*. <br/>

O arquivo usado na primeira parte foi o **[dataAll.txt](https://github.com/adhamlucas/supervised-learning-pp2/blob/master/rna-2020.1-pp2-data/dataAll.txt)**.<br/>

Na segunda parte, o arquivo **[data2.txt](https://github.com/adhamlucas/supervised-learning-pp2/blob/master/rna-2020.1-pp2-data/data2.txt)** foi utilizado, pois o identificador da equipe é o número 2.<br/>

O arquivo **[dataHoldout.txt](https://github.com/adhamlucas/supervised-learning-pp2/blob/master/rna-2020.1-pp2-data/dataHoldout.txt)** foi utilizado na terceira parte.

### Executando o projeto
No terminal, execute os seguintes comandos dentro de um diretório de sua preferência

```
$ git clone https://github.com/adhamlucas/E supervised-learning-pp2.git

$ cd supervised-learning-pp2

$ jupyter notebook
```

---

## Tecnologias
  - **[Anaconda](https://www.anaconda.com/products/individual)**
  - **[Jupyter Notebook](https://jupyter.org/)**


---

## Ellolib
  ```python
  class Perceptron (self, dataset, weight_random_seed=[-0.5, 0.5], bias=-1, degree=0, learn_tax=0.1, split="without")
  ```
| Parameters         | Description                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------- |
| dataset            | **{array-like, numpy-array}** </br> Training Data                                                              |
| weight_random_seed | **{array-like, numpy-array}, default=[-0.5, 0.5]** </br> Weight array like [w0, w1] to init and fit perceptron |
| bias               | **{float}, default=-1** </br> Number used in perceptron model to make the converge                             |
| degree             | **{float}, default=0** </br> Used like limiar in step function                                                 |
| learn_tax          | **{float}, default=0.1** </br> Value used to training perceptron that can determine the speed of the training  |
| split              | **{'holdout', 'without'}, default='without'** </br> Value to determine the method of split the dataset         |

<br/>

---

<br/>

| Methods                                          | Description                                                                                                                                                                                                          | Returns                                                                                                       |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| fit()                                            | Method used to fit the perceptron                                                                                                                                                                                    | none                                                                                                          |
| fit2(epoch)                                      | epoch:**{int}** <br/>  Method used to fit the perceptron with a specif number of epochs                                                                                                                              | none                                                                                                          |
| predict()                                        | Method used to fit the perceptron with a specific number of epochs                                                                                                                                                   | results: **{ndarray}** <br/> Return results of the predict made with the test dataset                         |
| get_confusion_matrix(reais, preditos, labels)    | reais: **{ndarray} of shape(n_samples, 2)** <br/> preditos: **{ndarray} of shape(n_samples, 1)** <br/> labels: **{array or ndarray} of shape(1, 2)** <br/> Get the confunsion matrix of the model trained and tested | X: **{ndarray} of shape(2, 2)** <br/> Return the confusion matrix of the model                                |
| get_accuracy(confusion_matrix)                   | confusion_matrix:**{ndarray, shape(2,2)}** <br/> Get the accuracy metric                                                                                                                                             | X: **{float}** <br/> Return the accuracy metric of the model extracted from the confusion matrix              |
| get_precision(confusion_matrix)                  | confusion_matrix:**{ndarray, shape(2,2)}** <br/> Get the precision metric                                                                                                                                            | X: **{float}** <br/> Return the precision metric of the model calculated from the confusion matrix            |
| get_recall(confusion_matrix)                     | get_reacall: **{ndarray}, shape(2, 2)**                                                                                                                                                                              | X:**{float}** <br/> returns the proportion of positives correctly identified                                  |
| get_f_score(precision, recall)                   | get_f_score: **{float, float}**                                                                                                                                                                                      | X: **{float}**<br/> Returns the balance between accuracy and model recall                                     |
| activation_function(value)                       | activation_function: **{float}**                                                                                                                                                                                     | X: **{int}**<br/>Perceptron activation function                                                               |
| adjust_weights(x_enter, result, expected_result) | adjust_weights: **{Union, int, expected_result}**                                                                                                                                                                    | X: **{float}**<br/>                                                        returns the adjusted weights       |
| generate_weights(weight_random_seed)             | generate_weights: **{array}**                                                                                                                                                                                        | X: **{list}**<br/>returns the weights generated                                                               |
| generate_train_test_dataset(split)               | generate_train_test_dataset:**{str}**                                                                                                                                                                                | X: **{tuple}**<br/>returns the separation of training and testing of the dataset, which can be holdout or not |
| generate_hyperplane()                            | generate_hyperplane: **{}**                                                                                                                                                                                          | X: **{tuple}**<br/>returns the points of the hyperplane line                                                  |






  

---

## Autores

**[Adham Lucas](https://github.com/adhamlucas)**<br/>
**[Enrique Izel](https://github.com/EnriqueIzel2)**<br/>
**[Nayara Cerdeira](https://github.com/NayaraCerdeira)**<br/>
**[Vitor Simões](https://github.com/VitorSimoes)**
