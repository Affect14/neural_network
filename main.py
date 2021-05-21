import inline
import matplotlib

import matplotlib.pyplot as plt
import numpy
import scipy.special


class neuralNetwork:
    # инициализация нейронки
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # матрицы весовых коэффициентов связкей wih и who
        # весовые коэффициенты связей между узлом i и узлом j
        # следующего слоя обозначены как w_i_j:
        # w11 w21
        # w12 w22 и т.д.
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        self.lr = learningrate  # коэф обучения
        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # тренировка нейронки
    def train(self, inputs_list, targets_list):
        # преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибки выходного слоя = (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты для связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # обновить весовые коэффициенты для связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # опрос нейронки
    def query(self, inputs_list):
        # преобразовать список входных значений
        # в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# print(n.query([1.0, 0.5, -1.5]))

test_data_file = open("mnist_dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 2
for e in range(epochs):
    # тренировка нейронной сети
    for record in training_data_list:
        # получить список значений, используя символы запятой (',') в качестве разделителей
        all_values = record.split(',')
        # масштабировать и сместить входные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # создать целевые выходные значения (все равны 0,01 за исключением желаемого маркерного значения, равного 0,99)
        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass

# all_values = test_data_list[0].split(',')
# print(all_values[0])
#
# image_array = numpy.asfarray(all_values[1:]).reshape(28, 28)
# matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
# plt.show()
# n.query((numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01)

# Журнал оценок работы сети, первоначально пустой
scorecard = []

# перебрать все записи в тестовом наборе данных
for record_2 in test_data_list:
    # получить список значений из записи, использкя символы запятой в качестве разделителя
    all_values_2 = record_2.split(',')
    # правильный ответ - первое значение
    correct_label = int(all_values_2[0])
    # print(correct_label, "истинный маркер")
    # масштабировать и сместить входные значения
    inputs = (numpy.asfarray(all_values_2[1:]) / 255.0 * 0.99) + 0.01
    # опрос сети
    outputs = n.query(inputs)
    # индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)
    # print(label, "ответ сети")
    # присоединить оценку ответа сети к концу списка
    if (label == correct_label):
        # в случае правильного ответа сети присоединить к списку значение 1
        scorecard.append(1)
    else:
        # в случае неправильного присоединить значение 0
        scorecard.append(0)
        pass
    pass

# рассчитать показатель эффективности в виде доли правильных ответов
print(scorecard)
scorecard_array = numpy.asarray(scorecard)
print("эффектиновсть =", scorecard_array.sum() / scorecard_array.size)