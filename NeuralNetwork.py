import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        '''
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        '''
        if len(layers) < 2:
            raise ValueError("layers parameter need to have at least 2 values")

        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        '''
        weights are initialized to be a random number in [-0.25, 0.25]
        all layers except the last one have one extra element - 
        bias unit which cor­re­sponds to the threshold value for the activation
        weights between layer i with n units and a layer j with m units = 
        a matrix n x m, each row = weights of one of i unit to all j units
        row = wij, j = 1..m
        '''
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
        # last element has no bias unit
        self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    # Stochastic Gradient Decent training
    def fit_sgd(self, X, y, learning_rate=0.2, epochs=10000):
        '''
        :param X: input vector
        :param y: target
        :param learning_rate: how much the weights will change in proportion 
        to error
        :param epochs: number of iterations
        '''
        # stochastic gradient descent, which chooses randomly a sample from
        # the training data and does the back­prop­a­ga­tion for that sample, and
        # this is repeated for a number of times (called epochs)
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
