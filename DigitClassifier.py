import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
from NetworkConfigHandler import NeuralNetworkConfig
from DataProcessor import DataPipeline
import matplotlib.pyplot as plt


def main():
    mnist_test = 'mnist_test.csv'
    mnist_train = 'mnist_train.csv'

    # get datasets
    delimeter = ','
    labels = ['id'] + [f'pixel_{i}' for i in range(784)]

    train_df = DataPipeline.load_data(mnist_train, delimeter, labels)
    test_set = DataPipeline.load_data(mnist_test, delimeter, labels)
    
    # decrease dataset size
    # train_df = train_df[0 : int(train_df.shape[0]*.1)][:]
    # test_set = test_set[0 : int(train_df.shape[0]*.1)][:]

    # split training data into training & validation sets
    split = 0.2
    train_set, valid_set = DataPipeline.generate_validation_set(train_df, split)
    
    # construct input output vector sets
    set_splits = lambda digit_set: (digit_set.iloc[:, 1:].to_numpy(), digit_set.iloc[:, 0].to_numpy())
    X_train, y_train = set_splits(train_set)
    X_valid, y_valid = set_splits(valid_set)
    X_test, y_test = set_splits(test_set)

    # normalize input data
    X_train = X_train / 255
    X_valid = X_valid / 255
    X_test = X_test / 255

    # initialise neural network parameters
    n = train_set.shape[1] - 1
    layer_config = [n, 784, 784, 784, 10]
    alpha = 12
    activation_function = "leaky_ReLU"
    const_c = 0.1
    
    # construct neural network
    NN = NeuralNetwork(n, alpha, layer_dimensions=layer_config, activation_func=activation_function, c=const_c)
    
    #train neural network
    epochs = 4
    NN.train(X_train, y_train, X_valid, y_valid, X_test, y_test, epochs)
    NN.evaluate(X_test, y_test)
    
    # store Neural Network data here
    # ATTENTION - take note of the store and load methods
    model_name = 'model5'
    NeuralNetworkConfig.store_network_config(NN, model_name)
    
    return

if __name__ == '__main__':
    main()