#!/usr/bin/env python
"""
Gradient Descent Optimization Algorithms

This module implements various gradient descent optimizers from scratch:
- Vanilla GD: Basic gradient descent with fixed learning rate
- Momentum: Accelerates GD by accumulating velocity in consistent directions
- Nesterov: "Look-ahead" momentum that computes gradient at predicted position
- RMSprop: Adapts learning rate per-parameter based on gradient magnitude
- Adam: Combines Momentum + RMSprop (most widely used in practice)

References:
- https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
- http://ruder.io/optimizing-gradient-descent/index.html#nesterovacceleratedgradient
- http://ruder.io/optimizing-gradient-descent/
- https://zhuanlan.zhihu.com/p/31630368
- Kingma & Ba, 2014: "Adam: A Method for Stochastic Optimization"
"""
import numpy as np
from const import (
    NO_MOMENTUM,
    MOMENTUM,
    NESTEROV,
    RMSPROP,
    ADAM
)


class Gradient(object):
    """Optimizer class that applies gradient updates to parameters."""

    def __init__(self, config):
        # Config stores hyperparameters (lr, beta1, beta2) and optimizer state (m, v, t)
        self.config = config

    def wrap(self, x, dx, key):
        """Route to the correct optimizer based on config type."""
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
        """
        Vanilla Gradient Descent: x = x - lr * gradient
        Simple but can be slow and oscillate in ravines.
        """
        _ = key
        learning_rate = self.config['learning_rate']
        x = x - learning_rate * dx
        return x

    def gd_momentum(self, x, dx, key):
        """
        Momentum: Accumulates velocity to accelerate in consistent directions.
        v = momentum * v - lr * gradient
        x = x + v

        Like a ball rolling downhill - builds up speed in consistent directions.

        :param x: parameter value
        :param dx: gradient
        :param key: options for ["W1", "W2", "W3", "b1", "b2", "b3"]
        :return: updated parameter
        """
        learning_rate = self.config['learning_rate']
        momentum = self.config['momentum']
        v = self.config.get('velocity.%s' % key, np.zeros_like(x))
        v = momentum * v - learning_rate * dx
        self.config['velocity.%s' % key] = v
        return x + v

    def nesterov(self, x, dx, key):
        """
        Nesterov Accelerated Gradient: "Look-ahead" momentum.
        Another momentum algorithm that computes gradient at the predicted future position.
        This gives a more accurate gradient and often converges faster than standard momentum.
        """
        learning_rate = self.config['learning_rate']
        momentum = self.config['momentum']
        v = self.config.get('velocity.%s' % key, np.zeros_like(x))
        v = momentum * v - learning_rate * dx
        self.config['velocity.%s' % key] = v
        cur_x = x + v
        # predict next x
        next_x = cur_x + momentum * v  # Look ahead to predicted position
        return next_x

    def rmsprop(self, x, dx, key):
        """
        RMSprop: Adapts learning rate per-parameter.
        Divides learning rate by running average of gradient magnitudes.
        Parameters with large gradients get smaller updates (prevents overshooting).
        Parameters with small gradients get larger updates (speeds up learning).
        """
        learning_rate = self.config['learning_rate']
        decay_rate = self.config['decay_rate']
        epsilon = self.config['epsilon']  # Prevents division by zero
        cache = self.config.get('cache.%s' % key, np.zeros_like(x))

        # Exponential moving average of squared gradients
        self.config['cache.%s' % key] = decay_rate * cache + (1 - decay_rate) * (dx ** 2)
        dx = learning_rate * dx / (np.sqrt(self.config['cache.%s' % key]) + epsilon)
        next_x = x - dx

        return next_x

    def adam(self, x, dx, key):
        """
        Adam: Adaptive Moment Estimation (Momentum + RMSprop combined).

        m = beta1 * m + (1 - beta1) * gradient      # First moment (momentum)
        v = beta2 * v + (1 - beta2) * gradient^2    # Second moment (RMSprop)
        x = x - lr * m / sqrt(v)                    # Update rule

        Bias correction (dividing by 1-beta^t) compensates for zero initialization
        of m and v, especially important in early iterations.
        """
        learning_rate = self.config['learning_rate']
        beta1 = self.config['beta1']  # Decay rate for first moment (default: 0.9)
        beta2 = self.config['beta2']  # Decay rate for second moment (default: 0.999)
        epsilon = self.config['epsilon']  # Numerical stability (default: 1e-8)

        # Get stored state or initialize to zeros
        m = self.config.get('m.%s' % key, np.zeros_like(x))  # First moment
        v = self.config.get('v.%s' % key, np.zeros_like(x))  # Second moment
        t = self.config.get('t.%s' % key, 0)  # Timestep counter

        # Update timestep and moments
        self.config['t.%s' % key] = t + 1
        self.config['m.%s' % key] = beta1 * m + (1 - beta1) * dx
        self.config['v.%s' % key] = beta2 * v + (1 - beta2) * (dx ** 2)

        # Bias correction: compensate for zero initialization
        mt = m / (1 - beta1 ** self.config['t.%s' % key])
        vt = self.config['v.%s' % key] / (1 - beta2 ** self.config['t.%s' % key])

        # Final update: momentum direction, scaled by adaptive learning rate
        next_x = x - learning_rate * mt / (np.sqrt(vt) + epsilon)

        return next_x
