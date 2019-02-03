#!/usr/bin/env python
import util
import numpy as np
import json
from gradient import Gradient
import matplotlib.pyplot as plt
import datetime

# https://www.tensorflow.org/tutorials/keras/basic_classification 
from const import (
    NO_MOMENTUM,
    MOMENTUM,
    NESTEROV,
    RMSPROP,
    ADAM,
    CONFIG,
    ITERATIONS
)


# https://github.com/NISH1001/naive-ocr/blob/009c99488d/ann.py
# https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions
# https://www.tensorflow.org/tutorials/keras/basic_classification
class Starter(object):
    """
    hidden nodes: 500, 100, last layer nodes: 10
    activation function: sigmoid, sigmoid, softmax
    loss function: crossentropy
    """
    def __init__(self, train_images, train_labels, iterations=1000, config={}, layer1=500, layer2=100, N=60000, M=10000):
        self.w_train_images = train_images
        self.w_train_labels = train_labels
        self.train_images = None
        self.train_labels = None
        self.iterations = iterations
        self.gradient = Gradient(config)
        self.layer1 = layer1
        self.layer2 = layer2
        self.accs_train = []
        self.loss = []
        self.result = []

    def init_parameters(self):
        np.random.seed(0)
        return {
            "W1": np.random.randn(self.layer1, 784) * 0.1,
            "b1": np.random.randn(self.layer1, 1) * 0.1,
            "W2": np.random.randn(self.layer2, self.layer1) * 0.1,
            "b2": np.random.randn(self.layer2, 1) * 0.1,
            "W3": np.random.randn(10, self.layer2) * 0.1,
            "b3": np.random.randn(10, 1) * 0.1
        }

    @staticmethod
    def forward_propagation(X, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]

        Z1 = np.dot(W1, X) + b1
        A1 = util.sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = util.sigmoid(Z2)
        Z3 = np.dot(W3, A2) + b3
        A3 = util.softmax(Z3)

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2,
                 "Z3": Z3,
                 "A3": A3}
        return A3, cache

    def update_parameters(self, parameters, grads):
        # Update rule for each parameter
        if self.gradient.config['type'] not in [NO_MOMENTUM, MOMENTUM, NESTEROV, RMSPROP, ADAM]:
            raise Exception("No type")

        return {"W1": self.gradient.wrap(parameters["W1"], grads["dW1"], "W1"),
                "b1": self.gradient.wrap(parameters["b1"], grads["db1"], "b1"),
                "W2": self.gradient.wrap(parameters["W2"], grads["dW2"], "W2"),
                "b2": self.gradient.wrap(parameters["b2"], grads["db2"], "b2"),
                "W3": self.gradient.wrap(parameters["W3"], grads["dW3"], "W3"),
                "b3": self.gradient.wrap(parameters["b3"], grads["db3"], "b3")}

    @staticmethod
    def softmax_cross_entropy_loss_der(A3, Y):
        dZ = np.array(A3, copy=True)
        n, m = dZ.shape
        for i in range(0, m):
            dZ[int(Y[0][i]), i] -= 1
        return dZ/m

    def backward_propagation(self, parameters, cache):
        # https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/02-Improving-Deep-Neural-Networks/week1/Programming-Assignments/Regularization/reg_utils.py
        # https://www.tensorflow.org/tutorials/keras/basic_classification
        # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
        dL_Z3 = Starter.softmax_cross_entropy_loss_der(cache['A3'], self.train_labels)
        dW3 = np.dot(dL_Z3, cache['A2'].T)
        db3 = np.sum(dL_Z3, axis=1, keepdims=True)

        dL_A2 = np.dot(parameters['W3'].T, dL_Z3)
        dL_Z2 = np.multiply(dL_A2, util.sigmoid_der(cache['Z2']))
        dW2 = np.dot(dL_Z2, cache['A1'].T)
        db2 = np.sum(dL_Z2, axis=1, keepdims=True)
        dL_A1 = np.dot(parameters['W2'].T, dL_Z2)
        dL_Z1 = np.multiply(dL_A1, util.sigmoid_der(cache['Z1']))
        dW1 = np.dot(dL_Z1, self.train_images.T)
        db1 = np.sum(dL_Z1, axis=1, keepdims=True)

        grads = {
            "dW3": dW3,
            "db3": db3,
            "dW2": dW2,
            "db2": db2,
            "dW1": dW1,
            "db1": db1,
        }
        return grads

    @staticmethod
    def classify(X):
        # find the max value of each prediction
        Ypred = []
        n, m = X.shape
        for i in range(0, m):
            max_value = -1
            index = -1
            for j in range(0, n):
                if X[j, i] > max_value:
                    max_value = X[j, i]
                    index = j
            if index != -1:
                Ypred.append(index)
        return Ypred

    @staticmethod
    def get_accuracy_rate(X, Y):
        pred = Starter.classify(X)
        tr = Starter.accuracy(pred, Y)
        return tr / (len(pred) + 0.0)

    @staticmethod
    def accuracy(pred, true_label):
        count = 0
        true_label = true_label.reshape([true_label.shape[1], 1]).tolist()
        for i in range(0, len(pred)):
            if pred[i] == true_label[i][0]:
                count += 1
        return count

    @staticmethod
    def chunks(ins, chunk_size):
        for i in range(0, len(ins), chunk_size):
            yield ins[i:i + chunk_size]

    def train(self):
        # init parameters
        parameters = self.init_parameters()
        for ii in range(self.iterations):
            pre_idx = 0
            for index in range(100, N, 1000):
                self.train_images = self.w_train_images[:, pre_idx:index]
                self.train_labels = self.w_train_labels[:, pre_idx:index]
                pre_idx = index
                Y, cache = Starter.forward_propagation(self.train_images, parameters)
                accuracy = Starter.get_accuracy_rate(Y, self.train_labels)
                loss = util.cross_entropy_loss(Y, self.train_labels)
                grads = self.backward_propagation(parameters, cache)
                parameters = self.update_parameters(parameters, grads)

            self.accs_train.append(accuracy)
            self.loss.append(loss)
            rx = "iterations: %s, accuracy: %s, time: %s" % ((ii + 1), accuracy, datetime.datetime.now())
            print(rx)
            self.result.append(rx)
        return parameters


def shuffle(inputs, num):
    values = util.get_shuffle_values(int(num))
    ans = np.ndarray(shape=inputs.shape)
    for idx in range(num):
        ans[idx] = inputs[values[idx]]
    return ans


if __name__ == "__main__":
    print("start loading the data...")
    train_images, train_labels, test_images, test_labels = util.load_data()

    # for test
    # train_images, train_labels, test_images, test_labels = train_images[:10000], train_labels[:10000], test_images, test_labels
    # ITERATIONS = 10

    util.plot_origin_image(5, 5, train_images, train_labels)  # plot original images
    print("start proprocessing the data...")
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    N = train_images.__len__()  # length of train dataset 
    M = test_images.__len__()  # length of test dataset

    train_images, train_labels, test_images, test_labels = shuffle(train_images, N), shuffle(train_labels, N), shuffle(test_images, M), shuffle(test_labels, M)

    t_train_images = train_images.reshape((N, 28 * 28)).astype(float).T
    t_test_images = test_images.reshape((M, 28 * 28)).astype(float).T
    t_train_labels = train_labels.reshape((N, 1)).T
    t_test_labels = test_labels.reshape((M, 1)).T

    print("start training the data...")
    obj = Starter(train_images=t_train_images,
                  train_labels=t_train_labels,
                  iterations=ITERATIONS,
                  config=CONFIG,
                  N = N,
                  M = M)
    parameters = obj.train()
    print("end training the data...")

    predictions, cache = Starter.forward_propagation(t_test_images, parameters)
    accuracy = obj.get_accuracy_rate(predictions, t_test_labels)
    logs = "test data set accuracy: %s" % accuracy
    print(logs)
    obj.result.append(logs)

    # first 15 images with predictions
    num_rows, num_cols = 5, 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        util.plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        util.plot_value_array(i, predictions, test_labels)
    plt.show()

    # plot the accuracy curve
    x, y = [], []
    for idx, values in enumerate(obj.accs_train):
        x.append(idx + 1)
        y.append(values * 100)

    plt.plot(x, y, label="accuracy for training data set")
    leg = plt.legend()
    leg.get_frame().set_alpha(0.5)
    plt.xlabel('iterations')
    plt.ylabel('accuracy %')
    plt.show()

    # plot the loss curve
    x, y = [], []
    for idx, values in enumerate(obj.loss):
        x.append(idx + 1)
        y.append(values)

    plt.plot(x, y, label="loss for training data set")
    leg = plt.legend()
    leg.get_frame().set_alpha(0.5)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

    # writing the corresponding log data into files
    with open("./plots_data/loss_%s_(iteration:%s).dat" % (CONFIG['name'].lower(), ITERATIONS), "w+") as writer:
        writer.write(json.dumps({"line_name": CONFIG['name'].lower(), "values": obj.loss}))

    with open("./plots_data/accuracy_%s_(iteration:%s).dat" % (CONFIG['name'].lower(), ITERATIONS), "w+") as writer:
        writer.write(json.dumps({"line_name": CONFIG['name'], "values": obj.accs_train}))

    with open("./plots_data/log_%s_(iteration:%s).result" % (CONFIG['name'].lower(), ITERATIONS), "w+") as writer:
        writer.write("\n".join(obj.result))

