#!/usr/bin/env python
from os.path import abspath, dirname
BASE_PATH = abspath(dirname(abspath(__file__)))

# kinds of gradient descent algorithms
NO_MOMENTUM = 0
MOMENTUM = 1
NESTEROV = 2
RMSPROP = 3
ADAM = 4

CLASS_NAMES = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']

# the iterating times
ITERATIONS = 100

# config
config01 = {
    "type": NO_MOMENTUM,
    "name": "NO_MOMENTUM",
    "learning_rate": 0.1
}

config02 = {
    "type": MOMENTUM,
    "name": "MOMENTUM",
    "learning_rate": 0.1,
    "momentum": 0.9
}

config03 = {
    "type": NESTEROV,
    "name": "NESTEROV",
    "learning_rate": 0.1,
    "momentum": 0.9
}

config04 = {
    "type": RMSPROP,
    "name": "RMSPROP",
    "learning_rate": 0.01,
    "decay_rate": 0.9,
    "epsilon": 1e-8
}

config05 = {
    "type": ADAM,
    "name": "ADAM",
    "learning_rate": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8
}

# change it to use different descent algorithms
CONFIG = config01
