#!/usr/bin/env python
import numpy as np
from const import (
    NO_MOMENTUM,
    MOMENTUM,
    NESTEROV,
    RMSPROP,
    ADAM
)


# https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
# http://ruder.io/optimizing-gradient-descent/index.html#nesterovacceleratedgradient
# http://ruder.io/optimizing-gradient-descent/
# https://zhuanlan.zhihu.com/p/31630368
class Gradient(object):
    def __init__(self, config):
        self.config = config

    def wrap(self, x, dx, key):
        if self.config['type'] == NO_MOMENTUM:
            return self.gd(x, dx, key)
        elif self.config['type'] == MOMENTUM:
            return self.gd_momentum(x, dx, key)
        elif self.config['type'] == NESTEROV:
            return self.nesterov(x, dx, key)
        elif self.config['type'] == RMSPROP:
            return self.rmsprop(x, dx, key)
        elif self.config['type'] == ADAM:
            return self.adam(x, dx, key)
        else:
            raise Exception("no function")

    def gd(self, x, dx, key):
        _ = key
        learning_rate = self.config['learning_rate']
        x = x - learning_rate * dx
        return x

    def gd_momentum(self, x, dx, key):
        """
        :param x:
        :param dx:
        :param key: options for ["W1", "W2", "W3", "b1", "b2", "b3"]
        :return:
        """
        learning_rate = self.config['learning_rate']
        momentum = self.config['momentum']
        v = self.config.get('velocity.%s' % key, np.zeros_like(x))
        v = momentum * v - learning_rate * dx
        self.config['velocity.%s' % key] = v
        return x + v

    def nesterov(self, x, dx, key):
        # another momentum algorithm
        learning_rate = self.config['learning_rate']
        momentum = self.config['momentum']
        v = self.config.get('velocity.%s' % key, np.zeros_like(x))
        v = momentum * v - learning_rate * dx
        self.config['velocity.%s' % key] = v
        cur_x = x + v
        # predict next x
        next_x = cur_x + momentum * v
        return next_x

    def rmsprop(self, x, dx, key):
        learning_rate = self.config['learning_rate']
        decay_rate = self.config['decay_rate']
        epsilon = self.config['epsilon']
        cache = self.config.get('cache.%s' % key, np.zeros_like(x))

        self.config['cache.%s' % key] = decay_rate * cache + (1 - decay_rate) * (dx ** 2)
        dx = learning_rate * dx / (np.sqrt(self.config['cache.%s' % key]) + epsilon)
        next_x = x - dx

        return next_x

    def adam(self, x, dx, key):
        learning_rate = self.config['learning_rate']
        beta1 = self.config['beta1']
        beta2 = self.config['beta2']
        epsilon = self.config['epsilon']

        m = self.config.get('m.%s' % key, np.zeros_like(x))
        v = self.config.get('v.%s' % key, np.zeros_like(x))
        t = self.config.get('t.%s' % key, 0)

        self.config['t.%s' % key] = t + 1
        self.config['m.%s' % key] = beta1 * m + (1 - beta1) * dx
        self.config['v.%s' % key] = beta2 * v + (1 - beta2) * (dx ** 2)

        mt = m / (1 - beta1 ** self.config['t.%s' % key])
        vt = self.config['v.%s' % key] / (1 - beta2 ** self.config['t.%s' % key])

        next_x = x - learning_rate * mt / (np.sqrt(vt) + epsilon)

        return next_x
