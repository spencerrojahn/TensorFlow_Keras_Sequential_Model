# TensorFlow_Keras_Sequential_Model
Uses TensorFlow Keras' Sequential model API to develop a 2-layer neural network that can predict cancer type (benign or malignant). 
- Thus, you must have the TensorFlow python module (*tensorflow*) installed on your machine.

How to run program (command): "python3 generate_model.py" 

The "breast-cancer-wisconsin-INFO.txt" file contains information on the breast cancer dataset.
The "breast-cancer-wisconsin.txt" file contains the breast cancer data.

This program generates, trains and evaluates a TensorFlow Keras Sequential model with two dense (fully-connected) layers comprised of 30 and 10 nodes, respectively. The model comes close to 100% accuracy, but it tops out around 94-97% when trained with max epochs set to 5. Thus, the training is done very quickly, so the accuracy is not as high. In order to get better accuracy, train with more epochs.

