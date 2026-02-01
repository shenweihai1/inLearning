# Neural Network from Scratch: Gradient Descent Optimization

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Neural networks implemented **from scratch without any deep learning libraries** (TensorFlow, Keras, PyTorch, etc.) to classify the Fashion-MNIST dataset. This project includes both a **fully-connected network** and a **Convolutional Neural Network (CNN)** achieving **95%+ accuracy**.


## Features

- Pure NumPy implementation - **no deep learning frameworks**
- **Two network architectures:**
  - Fully-connected network
  - CNN
- Two gradient descent optimization algorithms:
  - Vanilla Gradient Descent (No Momentum)
  - Adam (recommended)
- CNN components implemented from scratch:
  - Conv2D with im2col optimization
  - MaxPooling
  - Batch Normalization
  - Dropout
  - Flatten
- Fashion-MNIST dataset classification (10 clothing categories)

## Installation

```bash
# Clone the repository
git clone https://github.com/shenweihai1/inLearning.git
cd inLearning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install numpy scipy matplotlib
```

### Dataset Setup

Download the Fashion-MNIST dataset files and place them in the `dataset/` directory:

| File | Description |
|------|-------------|
| `train-images-idx3-ubyte.gz` | Training set images (60,000) |
| `train-labels-idx1-ubyte.gz` | Training set labels |
| `t10k-images-idx3-ubyte.gz` | Test set images (10,000) |
| `t10k-labels-idx1-ubyte.gz` | Test set labels |

Download from: [Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)

## Quick Start

### Configuration

Use command-line arguments to customize training:

**Fully-Connected Network:**
```bash
# Train with Adam optimizer (default)
python starter.py --optimizer adam --iterations 50

# Train with Vanilla gradient descent
python starter.py --optimizer vanilla --iterations 50

# Plot saved results without training (Adam)
python starter.py --optimizer adam --iterations 50 --plot-only

# Plot saved results without training (Vanilla)
python starter.py --optimizer vanilla --iterations 50 --plot-only
```

**CNN:**
```bash
# Train CNN
python cnn_starter.py --epochs 30

# Plot saved results without training
python cnn_starter.py --plot-only
```

## Implementation Details

### Fully Connected Network

```
Input (784) → Dense(1024) → ReLU → Dropout → Dense(256) → ReLU → Dropout → Dense(10) → Softmax
```

| Layer | Nodes | Activation Function | Parameters |
|-------|-------|---------------------|------------|
| Input | 784 (28×28 pixels) | - | - |
| Hidden 1 | 1024 | ReLU + Dropout | W1: 1024×784, b1: 1024×1 |
| Hidden 2 | 256 | ReLU + Dropout | W2: 256×1024, b2: 256×1 |
| Output | 10 | Softmax | W3: 10×256, b3: 10×1 |

**Total Parameters:** 1,068,810 (~1.07M) | **Accuracy:** ~89-91%

### CNN Architecture (95%+ Accuracy)

```
Input (1, 28, 28)
    │
    ├── Conv2D(32, 3×3, pad=1) → BatchNorm → ReLU
    ├── Conv2D(32, 3×3, pad=1) → BatchNorm → ReLU
    ├── MaxPool(2×2)                                    → (32, 14, 14)
    │
    ├── Conv2D(64, 3×3, pad=1) → BatchNorm → ReLU
    ├── Conv2D(64, 3×3, pad=1) → BatchNorm → ReLU
    ├── MaxPool(2×2)                                    → (64, 7, 7)
    │
    ├── Flatten                                         → (3136,)
    ├── Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    └── Dense(10) → Softmax                             → (10,)
```

| Component | Details | Parameters |
|-----------|---------|------------|
| Conv1 | 32 filters, 3×3, pad=1 | 320 |
| Conv2 | 32 filters, 3×3, pad=1 | 9,248 |
| Conv3 | 64 filters, 3×3, pad=1 | 18,496 |
| Conv4 | 64 filters, 3×3, pad=1 | 36,928 |
| BatchNorm (×5) | Conv + FC layers | 768 |
| FC1 | 3136 → 256 | 803,072 |
| FC2 | 256 → 10 | 2,570 |
| **Total** | | **871,402** |

| Setting | Value |
|---------|-------|
| Kernel Size | 3×3 with padding=1 |
| Pooling | MaxPool 2×2, stride 2 |
| Regularization | BatchNorm + Dropout(0.5) |
| Optimizer | Adam (lr=0.001) |

## Performance Benchmarks


## References

### Papers
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - Kingma & Ba, 2014

### Online Resources
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

## License

MIT License