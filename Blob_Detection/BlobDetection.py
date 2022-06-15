# add path to source code
import sys
sys.path.append('./src/')
# import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from NeuralNetwork import NeuralNetwork
from NetworkConfigHandler import NetworkConfigHandler

def plot_clusters_2d(X_data, y_data):
    plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data)
    plt.show()

def plot_clusters_3d(data, labels):
    fig = plt.figure()
    ax = plt.axes(projection='3d', xlabel = 'x-axis', ylabel = 'y-axis', zlabel = 'z-axis')
    ax.set_facecolor('xkcd:grey')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
    plt.show()

def plot_model_hist(cost_history: dict, acc_history: dict) -> None:
    '''
    Plot cost data from a trained Neural Network.

    @params
    - cost_history: (dict) -> dictionary containing costs for all epochs from a trained Neural Network 

    @returns
    - None
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # get cost data for training and validation set
    cost_hist1 = cost_history['Train_cost']
    cost_hist2 = cost_history['Validation_cost']

    acc_hist1 = acc_history['Train_set']
    acc_hist2 = acc_history['Validation_set']
    epochs = [i+1 for i in range(len(cost_hist1))]
    
    # plot data
    ax1.plot(epochs, cost_hist1, label='Training Cost', color='blue')
    ax1.plot(epochs, cost_hist2, label='Validation Cost', color='green')
    ax1.set_title('Model Cost History')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Cost J')
    ax1.legend()

    ax2.plot(epochs, acc_hist1, label='Training Acc.', color='red')
    ax2.plot(epochs, acc_hist2, label='Validation Acc.', color='purple')
    ax2.set_title('Model Accuracy History')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.show()

    return

if __name__ == "__main__":
    # define file name containing blobs data
    data_file = 'blobs1'
    
    #create datasets
    data = pd.read_csv(f'./data/{data_file}.csv', )
    X, y = data.iloc[:, :-1].to_numpy(), data.loc[:, 'y'].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)
    
    # View the data
    plot_clusters_3d(X, y)
    
    # initialise neural network parameters
    n = X.shape[1]
    num_centers = int(max(y) + 1)

    layer_config = [n, num_centers]
    alpha = 0.001
    activation_function = "leaky_ReLU"
    const_c = 0.1
    
    # construct neural network
    model = NeuralNetwork(n, alpha, layer_dimensions=layer_config, activation_func=activation_function, c=const_c)
    
    #train neural network
    epochs = 4
    model.train(X_train, y_train, X_valid, y_valid, X_test, y_test, epochs)
    model.evaluate(X_test, y_test)
    plot_model_hist(model.cost_hist, model.accuracies)

    #----------------------------------------------------
    # store Neural Network data here
    # ATTENTION - take note of the store and load methods
    model_name = 'blob1'
    NetworkConfigHandler.store_network_config(model, f'./Blob_Detection/PreTrained_Blob_Models/{model_name}')