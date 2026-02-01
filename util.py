#!/usr/bin/env python
import gzip
import os
import json
import numpy as np
from scipy.special import expit
import scipy.special as sc
from const import BASE_PATH
from const import CLASS_NAMES
import matplotlib.pyplot as plt


# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.special.expit.html
def sigmoid(x):
    # return 1.0 / (1.0 + np.exp(-x))
    return expit(x)


def sigmoid_der(z):
    return sigmoid(z)*(1-sigmoid(z))


def relu(x):
    return np.maximum(0, x)


def relu_der(z):
    return (z > 0).astype(float)


def dropout(A, keep_prob):
    """Apply dropout to activation matrix A during training."""
    D = np.random.rand(*A.shape) < keep_prob
    A = np.multiply(A, D)
    A = A / keep_prob  # Scale to maintain expected value
    return A, D


def dropout_backward(dA, D, keep_prob):
    """Backward pass for dropout."""
    dA = np.multiply(dA, D)
    dA = dA / keep_prob
    return dA


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x - sc.logsumexp(x, axis=0))


def cross_entropy_loss(Z, Y):
    n, m = Z.shape
    A = np.zeros(Z.shape)
    for i in range(0, m):
        max_z = np.max(Z[:, i])
        col_sum = np.sum(np.exp(Z[:, i] - max_z))
        for j in range(0, n):
            A[j][i] = np.exp(Z[j][i] - max_z) / col_sum
    return softmax_entropy_loss(A, Y)


def softmax_entropy_loss(A, Y):
    n, m = A.shape
    eps = np.finfo(np.float32).eps
    loss = 0
    for i in range(0, m):
        loss += np.log(A[int(Y[0][i]), i] + eps)
    return loss / (-m)


def load_data():
    base = os.path.join(BASE_PATH, 'dataset')
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for file in files:
        paths.append(os.path.join(base, file))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return x_train, y_train, x_test, y_test


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[:,i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[:,i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    true_label = int(true_label)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
                                         100 * np.max(predictions_array),
                                         CLASS_NAMES[true_label]),
                                         color=color)


def plot_origin_image(m, n, train_images, train_labels):
    plt.figure(figsize=(10, 10))
    for i in range(m * n):
        plt.subplot(m, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(CLASS_NAMES[int(train_labels[i])])
    plt.show()


# def shuffle_values(num):
#     import random, json
#     ac = [[i] for i in range(num)]
#     random.shuffle(ac)
#     values = json.dumps(ac)
#     with open("%s" % num, "w+") as writer:
#         writer.write(values)

def get_shuffle_values(num):
    import json
    with open("dataset/%s.json" % num) as reader:
        lines = reader.readlines()
        line = lines[0]
        return json.loads(line)


def plot_by_file_names(ans, xlable=""):
    plt.figure(figsize=(16, 32))
    for f in ans:
        with open(f) as reader:
            p, q = [], []
            lines = reader.readlines()
            line = lines[0]
            values = json.loads(line)
            line_name, values = values['line_name'], values['values']
            for idx, v in enumerate(values):
                p.append(idx + 1)
                q.append(v)

            plt.plot(p, q, label=line_name)

    leg = plt.legend()
    leg.get_frame().set_alpha(0.5)
    plt.xlabel(xlable)
    plt.show()


if __name__ == "__main__":
    # a = np.ndarray(shape=(2, 5))
    # mx = softmax(a)
    # print(mx)

    # shuffle
    # shuffle_values(10000)
    #plot_by_file_names(["./plots_data/loss_adam_(iteration:200).dat",
    #                    "./plots_data/loss_momentum_(iteration:200).dat",
    #                    "./plots_data/loss_nesterov_(iteration:200).dat",
    #                    "./plots_data/loss_no_momentum_(iteration:200).dat",
    #                    "./plots_data/loss_rmsprop_(iteration:200).dat",
    #                    ])
    pass
