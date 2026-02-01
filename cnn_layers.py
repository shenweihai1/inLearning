#!/usr/bin/env python
"""
Convolutional Neural Network Layers
Implemented from scratch using NumPy
"""
import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Transform input images to column matrix for efficient convolution.

    Args:
        input_data: Input images (N, C, H, W)
        filter_h: Filter height
        filter_w: Filter width
        stride: Stride size
        pad: Padding size

    Returns:
        col: Column matrix
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Transform column matrix back to images.

    Args:
        col: Column matrix
        input_shape: Original input shape (N, C, H, W)
        filter_h: Filter height
        filter_w: Filter width
        stride: Stride size
        pad: Padding size

    Returns:
        img: Reconstructed images
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Conv2D:
    """
    2D Convolutional Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # He initialization
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)

        # Gradients
        self.dW = None
        self.db = None

        # Cache for backward pass
        self.x = None
        self.col = None
        self.col_W = None

    def forward(self, x):
        """
        Forward pass for convolution.

        Args:
            x: Input (N, C, H, W)

        Returns:
            out: Output (N, out_channels, out_H, out_W)
        """
        N, C, H, W = x.shape
        out_h = (H + 2 * self.pad - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.kernel_size) // self.stride + 1

        col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.pad)
        col_W = self.W.reshape(self.out_channels, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """
        Backward pass for convolution.

        Args:
            dout: Upstream gradient (N, out_channels, out_H, out_W)

        Returns:
            dx: Gradient w.r.t. input
        """
        N, C, H, W = self.x.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)

        return dx


class MaxPool2D:
    """
    Max Pooling Layer
    """
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

        self.x = None
        self.arg_max = None

    def forward(self, x):
        """
        Forward pass for max pooling.

        Args:
            x: Input (N, C, H, W)

        Returns:
            out: Output (N, C, out_H, out_W)
        """
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        col = im2col(x, self.pool_size, self.pool_size, self.stride, 0)
        col = col.reshape(-1, self.pool_size * self.pool_size)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        Backward pass for max pooling.

        Args:
            dout: Upstream gradient

        Returns:
            dx: Gradient w.r.t. input
        """
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_size * self.pool_size
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_size, self.pool_size, self.stride, 0)

        return dx


class Flatten:
    """
    Flatten layer to convert 4D tensor to 2D
    """
    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        """
        Forward pass: flatten input.

        Args:
            x: Input (N, C, H, W)

        Returns:
            out: Flattened output (N, C*H*W)
        """
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        """
        Backward pass: reshape gradient.

        Args:
            dout: Upstream gradient (N, C*H*W)

        Returns:
            dx: Reshaped gradient (N, C, H, W)
        """
        return dout.reshape(self.original_shape)


class Dense:
    """
    Fully connected layer
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # He initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input (N, in_features)

        Returns:
            out: Output (N, out_features)
        """
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        """
        Backward pass.

        Args:
            dout: Upstream gradient (N, out_features)

        Returns:
            dx: Gradient w.r.t. input
        """
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx


class ReLU:
    """
    ReLU activation layer
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Dropout:
    """
    Dropout layer
    """
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.mask = None
        self.training = True

    def forward(self, x, training=True):
        self.training = training
        if training:
            self.mask = np.random.rand(*x.shape) < self.keep_prob
            return x * self.mask / self.keep_prob
        else:
            return x

    def backward(self, dout):
        if self.training:
            return dout * self.mask / self.keep_prob
        else:
            return dout


class BatchNorm:
    """
    Batch Normalization layer
    """
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.momentum = momentum
        self.epsilon = epsilon

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            xc = x - mean
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + self.epsilon)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.std = std
            self.xn = xn

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + self.epsilon)

        return self.gamma * xn + self.beta

    def backward(self, dout):
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std ** 2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = np.sum(dout * self.xn, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        return dx


def softmax(x):
    """Numerically stable softmax"""
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """
    Cross entropy loss.

    Args:
        y_pred: Predictions (N, num_classes)
        y_true: True labels (N,) as integers

    Returns:
        loss: Scalar loss value
    """
    N = y_pred.shape[0]
    epsilon = 1e-7
    return -np.sum(np.log(y_pred[np.arange(N), y_true.astype(int)] + epsilon)) / N


def softmax_cross_entropy_backward(y_pred, y_true):
    """
    Backward pass for softmax + cross entropy.

    Args:
        y_pred: Predictions (N, num_classes)
        y_true: True labels (N,) as integers

    Returns:
        dout: Gradient (N, num_classes)
    """
    N = y_pred.shape[0]
    dout = y_pred.copy()
    dout[np.arange(N), y_true.astype(int)] -= 1
    return dout / N
