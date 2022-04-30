"""
Note to user: 

This script is used to analyse the performance of trained Neural Networks.
Feel free to play with the script and to add-on anything else as you wish.
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
from NeuralNetwork import NeuralNetwork
from NetworkConfigHandler import NeuralNetworkConfig
from DataPreProcessor import DataPreProcessor

def plot_cost_hist(cost_history: dict) -> None:
    '''
    Plot cost data from a trained Neural Network.

    @params
    - cost_history: (dict) -> dictionary containing costs for all epochs from a trained Neural Network 

    @returns
    - None
    '''
    # get cost data for training and validation set
    hist1 = cost_history['Train_cost']
    hist2 = cost_history['Validation_cost']
    epochs = [i+1 for i in range(len(hist1))]
    
    # plot data
    plt.plot(epochs, hist1, label='Training Cost', color='blue')
    plt.plot(epochs, hist2, label='Validation Cost', color='green')    
    plt.xlabel('epochs')
    plt.ylabel('Cost J')
    plt.legend()
    plt.title('Cost history plot')
    plt.show()

    return

def display_digits(neural_network, X_data, y_data, img_count):
    fig = plt.figure(figsize=(14, 14))

    for i in range(1, img_count + 1):
        img = np.reshape(X_data[i - 1], (np.shape(X_data[i - 1])[0], 1))
        id = y_data[i - 1]
        
        output = neural_network.forward_propagate(img)
        prediction = np.argmax(output, axis=0)

        prob = output[int(prediction)]

        pixels = img.reshape((28, 28))
        plt.subplot(4, 5, i)
        
        plt.imshow(pixels, cmap='gray')
        plt.title(f'Image Tag: {id}')
        plt.xlabel(f'NN prediction: {prediction}\nProbability: {round(prob[0] * 100, 3)}%') 

    plt.subplots_adjust(wspace=0.9, hspace=0.9)
    plt.show()

def compare_digits_classified(neural_network: NeuralNetwork, X_data, y_data):
    classes = {i: 0 for i in range(10)}
    for i in range(X_data.shape[0]):
        x = X_data[i][:]
        y_pred = neural_network.forward_propagate(x)
        
        if (np.argmax(y_pred) == y_data[i]):
            classes[y_data[i]] += 1
    class_elements, class_count = np.unique(y_data, return_counts=True)
    class_set = dict(zip(class_elements, class_count))

    plt.bar(class_set.keys(), class_set.values())
    plt.bar(classes.keys(), classes.values())
    plt.xticks(range(len(classes)), list(classes.keys()))
    for x, y in enumerate(classes.values()):
        plt.text(y, x, str(y))

    plt.show()

    return

def test_gradient(NN: NeuralNetwork, x, y, epsilon=1e-4):
    pred = NN.forward_propagate(x+epsilon)
    grad = NN.cost_gradient(pred, y)
    
    pred_prime = NN.forward_propagate(x - epsilon)
    grad_prime = NN.cost_gradient(pred_prime, y)
    estimate = (grad - grad_prime) / (2 * epsilon)
    return LA.norm(estimate)


def main():
    mnist_test = 'mnist_test.csv'
    mnist_train = 'mnist_train.csv'

    # get datasets
    delimeter = ','
    labels = ['id'] + [f'pixel_{i}' for i in range(784)]

    train_df = DataPreProcessor.load_data(mnist_train, delimeter, labels)
    test_set = DataPreProcessor.load_data(mnist_test, delimeter, labels)
    
    # decrease dataset size
    train_df = train_df[0 : int(train_df.shape[0]*.8)][:]
    test_set = test_set[0 : int(train_df.shape[0]*.8)][:]

    # split training data into training & validation sets
    split = 0.2
    train_set, valid_set = DataPreProcessor.split_dataset(train_df, split)
    
    # construct input output vector sets
    set_splits = lambda digit_set: (digit_set.iloc[:, 1:].to_numpy(), digit_set.iloc[:, 0].to_numpy())
    X_train, y_train = set_splits(train_set)
    X_valid, y_valid = set_splits(valid_set)
    X_test, y_test = set_splits(test_set)

    # normalize input data
    X_train = DataPreProcessor.normalize_dataset(X_train)
    X_valid = DataPreProcessor.normalize_dataset(X_valid)
    X_test = DataPreProcessor.normalize_dataset(X_test)

    #load data into neural network data structure
    model_name = 'PreTrained_Models/model6'
    NN = NeuralNetworkConfig.load_network_config(model_name)

    # test gradient
    # eps = 1e-10
    # sample_count = 20
    # average_error = 0
    # print('-------------------------------------------------')
    # print(f'Testing gradient with sample [epsilon = {eps}]:')
    # for i in range(sample_count):
    #     error_rate = 100 * test_gradient(NN, X_train[i], y_train[i], epsilon=eps)
    #     average_error += error_rate / sample_count

    # print(f'Average error rate in gradients computed: {round(average_error, 6)}%')
    # print('-------------------------------------------------')
    
    # plot cost data
    cost_hist = NN.cost_hist
    plot_cost_hist(cost_hist)
    # display_digits(NN, X_test, y_test, 20)
    # compare_digits_classified(NN, X_test, y_test)
    
    return

if __name__ == '__main__':
    main()