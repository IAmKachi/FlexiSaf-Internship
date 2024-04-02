import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

np.random.seed(24)


def plot_data(X, y):
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]

    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='blue', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='red', edgecolor='k')


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W) + b)[0])


def perceptronStep(X, y, W, b, learn_rate=0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)

        if y[i] - y_hat == 1:
            W[0] += X[i][0] * learn_rate
            W[1] += X[i][1] * learn_rate
            b += learn_rate
        elif y[i] - y_hat == -1:
            W[0] -= X[i][0] * learn_rate
            W[1] -= X[i][1] * learn_rate
            b -= learn_rate
        
        return W, b


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])

    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max

    # solution lines to plot
    boundary_lines = []
    for i in range(num_epochs):
        # in each epoch, we apply the perceptron step
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = pd.read_csv(r'week_2\input\data.csv', header=None)
X = np.array(data[[0, 1]])
y = np.array(data[2])

# boundary_lines = trainPerceptronAlgorithm(X, y)
# a = X[np.argwhere(y == 1)]
# print([s[0] for s in a])
# print(a)
# # print(boundary_lines)
# plot_data(X.T[0], X.T[1], y, boundary_lines=boundary_lines)
# plot_data(X, y)

W = np.array(np.random.rand(2, 1))
x_max = max(X.T[0])
b = np.random.rand(1)[0] + x_max

# plt.show()
# print(y * sigmoid(np.matmul(X, W) + b)
# print(X)
# print(np.matmul(X, W) + b)
print(np.matmul([ 0.78051,  -0.063669], [-0.53076476,  0.93080519]))
# a = np.array([2, 3])
# b = np.array([4, 5])
# print(a * b)