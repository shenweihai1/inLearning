# Neural Network from Scratch: Gradient Descent Optimization

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A fully-connected neural network implemented **from scratch without any deep learning libraries** (TensorFlow, PyTorch, etc.) to classify the Fashion-MNIST dataset.

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

```bash
# Train with Vanilla gradient descent (recommended)
python starter.py --optimizer vanilla --iterations 50

# Train with Adam optimizer
python starter.py --optimizer adam --iterations 150

# Plot saved results without training
python starter.py --optimizer vanilla --iterations 50 --plot-only

# Plot saved results without training
python starter.py --optimizer adam --iterations 150 --plot-only
```

## Network Architecture

```
Input (784) → Dense(1024) → ReLU → Dropout → Dense(256) → ReLU → Dropout → Dense(10) → Softmax
```

| Layer | Nodes | Activation Function | Parameters |
|-------|-------|---------------------|------------|
| Input | 784 (28×28 pixels) | - | - |
| Hidden 1 | 1024 | ReLU + Dropout | W1: 1024×784, b1: 1024×1 |
| Hidden 2 | 256 | ReLU + Dropout | W2: 256×1024, b2: 256×1 |
| Output | 10 | Softmax | W3: 10×256, b3: 10×1 |

**Total Parameters:** 1,068,810 (~1.07M)

## Performance Benchmarks

| Optimizer | Learning Rate | Iterations | Train Acc | Test Acc | Gap |
|-----------|---------------|------------|-----------|----------|-----|
| Vanilla | 0.1 | 50 | 90.7% | 88.19% | 2.5% |
| **Adam** | 0.001 | 150 | 97.0% | **89.43%** | 7.6% |

**Note:** Adam shows overfitting (7.6% gap). Possible fixes: fewer iterations, lower dropout (0.7), or add L2 regularization.

## References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) - Kingma & Ba, 2014
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)

## License

MIT License
