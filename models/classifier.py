# coding: utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
import torch

# function preprocessing data
def getMatrix(filename):
    all_data = torch.load(filename)
    y_list = []
    matrix_list = []

    dic = {1: 1, -1: 0}

    for item in all_data:
        y = item[1].data.numpy()
        sentence = np.array(item[0].data.numpy(),dtype=np.float64)
        y_list.append(dic[y[0]])
        count = 0
        matrix = []

        for word in sentence[0]:
            count = count+1

        for i in range(0,count):
            matrix.append(np.array(sentence[0][i]).reshape(1,768))

        for i in range(count,137):
            matrix.append(np.zeros([1,768],dtype=np.float64))

        matrix = np.array(matrix).reshape(1, 137 * 768)
        matrix_list.append(matrix[0])

    return matrix_list, y_list



#load data
mlist_train, ylist_train = getMatrix('train.data')
mlist_val, ylist_val = getMatrix('dev.data')
mlist_test, ylist_test = getMatrix('test.data')



#set hyperparameters
hidden_1 = 500
hidden_2 = 200
train_epoch = 20
trainbatch_size = 50


num_sen=len(mlist_train)
mlist_train=np.array(mlist_train).reshape(num_sen,137*768)
mlist_val=np.array(mlist_val).reshape(num_sen,137*768)
mlist_test=np.array(mlist_test).reshape(num_sen,137*768)



#model
model = keras.Sequential()
model.add(keras.layers.Dense(hidden_1, input_shape=(137*768,), activation=tf.nn.tanh))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(hidden_2, activation=tf.nn.tanh))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, activation=tf.nn.softmax))
print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])




x_val = mlist_val
partial_x_train = mlist_train
y_val = ylist_val
partial_y_train = ylist_train



model.fit(partial_x_train,
          partial_y_train,
          epochs=train_epoch,
          batch_size=trainbatch_size,
          validation_data=(x_val, y_val),
          shuffle=True)

print('---------------')
results = model.evaluate(mlist_test, ylist_test)
print(results)
