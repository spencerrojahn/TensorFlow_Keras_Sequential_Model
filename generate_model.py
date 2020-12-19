import os
# import sys
import numpy as np
import random
from tensorflow import keras

# ignore any messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# get suprvised data from breast cancer dataset
data_lines = open('breast-cancer-wisconsin.txt').read().splitlines()

x_lists = []
y = []

random.shuffle(data_lines)

for line in data_lines:

    if '?' not in line:

        entry = line.split(',')


        x = []
        for i in range(1, len(entry)-1):
            x.append(float(int(entry[i])/10))

        x_lists.append(x)
        y.append((int(entry[-1])-2)/2)

    else:
        continue


split_index = int(len(x_lists)*(3/4))

x_train_data = x_lists[:split_index]
y_train_data = y[:split_index]
x_test_data = x_lists[split_index:]
y_test_data = y[split_index:]

x_train = np.array(x_train_data, dtype=np.float16)
y_train = np.asarray(y_train_data, dtype=np.int8)
x_test = np.array(x_test_data, dtype=np.float16)
y_test = np.asarray(y_test_data, dtype=np.int8)


model = keras.Sequential(
    [
        keras.Input(shape=9),
        keras.layers.Dense(30, activation='relu', name='layer1'),       # Dense layers are fully-connected layers
        keras.layers.Dense(10, activation='relu', name='layer2'),
        keras.layers.Dense(2, name='layer_out'),       # output layer (10 outputs for MNIST dataset)
     
    ]
)


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)

print('\nTraining on breast cancer training data (random 75% of data):\n')
# train then model with the model.fit() function
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)

# evaluate the accuracy of the model
print('\nEvaluating on breast cancer testing data (other 25% of data):\n')
model.evaluate(x_test, y_test, verbose=1)

print('\n')


