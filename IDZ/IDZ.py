from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Устанавливаем воспроизводимость рандома
        random.seed(1)

        # Моделируем нейрон с тремя входами и одним выходом
        # Заполним матрицу 3 x 1 случайными весами от -1 до 1
        # и средним 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # Функция, описывающая S-образную кривую
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Производная функции
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Обучение нейронной сети
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Пропускаем обучающие данные через нейросеть
            output = self.think(training_set_inputs)

            # Вычисляем ошибку
            error = training_set_outputs - output

            # Умножаем ошибку на вход и на производную функции
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Регулируем нейронную сеть
            self.synaptic_weights += adjustment

    # Функция действия нейрона
    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    # Инициализируем нейронную сеть
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    # Обучающие данные
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Обучение нейронной сети
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("Новые веса после обучения: ")
    print (neural_network.synaptic_weights)

    # Тестирование работы нейронной сети в новой ситуации
    print ("Новая ситуация [1, 0, 0] -> ?: ")
    print (neural_network.think(array([1, 0, 0])))