### Gradient Descent Optimization
In this project, we implement a 3-layer fully-connected neural network(without any deep learning libraries like [TensorFlow](https://www.tensorflow.org/tutorials/keras/basic_classification), Keras,
MatConvNet etc.) to classify the fashion-MNIST dataset using No Momentum, Polyak’s classical momentum, Nesterov’s Accelerated Gradient, RmsProp and ADAM algorithms

### Neural network structure for the experiments
![x](https://raw.githubusercontent.com/shenweihai1/imageUrlService/master/inlearning/exp.png)

### Usage
Run `starter.py` as:
```
python starter.py
```
Configure `const.py` as:
```
try to repalce variable ITERATIONS with different iterations
try to repalce variable CONFIG with different variable config01 ~ config05 to experience different algorithm
```

### Training Result
#### Accuracy
![x](https://raw.githubusercontent.com/shenweihai1/imageUrlService/master/inlearning/acc.png)
#### Loss curve
![x](https://raw.githubusercontent.com/shenweihai1/imageUrlService/master/inlearning/loss.png)

