# add path to source code
import sys
sys.path.append('../src/')

# import required modules
import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
from NetworkConfigHandler import NetworkConfigHandler
from DataPreProcessor import DataPreProcessor


def main():
    mnist_test = './data/mnist_test.csv'
    mnist_train = './data/mnist_train.csv'

    # get datasets
    delimeter = ','
    labels = ['id'] + [f'pixel_{i}' for i in range(784)]

    train_set = DataPreProcessor.load_data(mnist_train, delimeter, labels)
    test_set = DataPreProcessor.load_data(mnist_test, delimeter, labels)
    
    # decrease dataset size
    train_set = train_set[0 : int(train_set.shape[0]*.1)][:]
    test_set = test_set[0 : int(train_set.shape[0]*.1)][:]

    # split test dataframe into test & validation sets
    split = 0.5
    test_set, valid_set = DataPreProcessor.split_dataset(test_set, split)
    
    # construct input output vector sets
    set_splits = lambda digit_set: (digit_set.iloc[:, 1:].to_numpy(), digit_set.iloc[:, 0].to_numpy())
    X_train, y_train = set_splits(train_set)
    X_valid, y_valid = set_splits(valid_set)
    X_test, y_test = set_splits(test_set)

    # normalize input data
    X_train = DataPreProcessor.normalize_dataset(X_train)
    X_valid = DataPreProcessor.normalize_dataset(X_valid)
    X_test = DataPreProcessor.normalize_dataset(X_test)

    # initialise neural network parameters
    n = train_set.shape[1] - 1
    layer_config = [n, 128, 64, 10]
    alpha = 20
    activation_function = "leaky_ReLU"
    const_c = 0.1
    
    # construct neural network
    NN = NeuralNetwork(n, alpha, layer_dimensions=layer_config, activation_func=activation_function, c=const_c)
    
    #train neural network
    epochs = 3
    NN.train(X_train, y_train, X_valid, y_valid, X_test, y_test, epochs)
    NN.evaluate(X_test, y_test)
    
    # store Neural Network data here
    # ATTENTION - take note of the store and load methods
    model_name = 'PreTrained_Models/model_test'
    NetworkConfigHandler.store_network_config(NN, model_name)
    
    return

if __name__ == '__main__':
    main()