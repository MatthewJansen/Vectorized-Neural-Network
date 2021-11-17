import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
from NetworkConfigHandler import NeuralNetworkConfig
import matplotlib.pyplot as plt


def load_data(filename: str, delimeter: str, labels: list):
    '''
    Used to load the data from a .csv file into a pandas DataFrame.

    @params
    - filename: (str) -> name of the .csv file containing all the data  
    - delimeter: (str) -> character(s) by which the data entries are seperated
    - labels: (list) -> labels for the columns in the DataFrame

    @returns
    - (pd.DataFrame) -> DataFrame containing data for the Neural Network
    '''
    return pd.read_csv(filename, sep=delimeter, names=labels)


def generate_validation_set(training_set: pd.DataFrame, split_ratio: float):
    '''
    Used to generate a validation dataset by means of extracting a ratio of
    data from the original training dataset.

    @params
    - training_set: (pd.DataFrame) -> original training dataset
    - split_ratio: (float) -> ratio of data which the validation set should receive
    
    @returns
    - new_train_set, valid_set: (tuple[pd.DataFrame, pd.DataFrame]) -> new training and validation datasets 

    Note:

    split_ratio is the size for the validation set, i.e. if the ratio is 0.2, then the 
    ratio going to the training set is 0.8 (80%) and the validation set will 0.2 (20%) 
    of the data.
    '''
    #get training set size
    n = training_set.shape[0]
    # copy and shuffle training set
    temp_set = training_set.sample(frac=1, random_state=42).reset_index(drop=True)
    #split data into respective ratios
    new_train_set = temp_set[0: int(n * (1 - split_ratio))][:]
    valid_set = temp_set[int(n * (1 - split_ratio)):][:]

    return new_train_set, valid_set


def plot_cost_hist(cost_history: dict) -> None:
    '''
    Plot cost data from a trained Neural Network.

    @params
    - cost_history: (dict) -> dictionary containing costs for all epochs from a trained Neural Network 

    @returns
    - None
    '''
    # get cost data for training and testing set
    hist1 = cost_history['Train_cost']
    hist2 = cost_history['Test_cost']
    epochs = [i+1 for i in range(len(hist1))]
    
    # plot data
    plt.plot(epochs, hist1, label='Training Cost', color='blue')
    plt.plot(epochs, hist2, label='Testing Cost', color='green')    
    plt.xlabel('epochs')
    plt.ylabel('Cost J')
    plt.legend()
    plt.title('Cost history plot')
    plt.show()

    return

def main():
    mnist_test = 'mnist_test.csv'
    mnist_train = 'mnist_train.csv'

    # get datasets
    delimeter = ','
    labels = ['id'] + [f'pixel_{i}' for i in range(784)]

    train_df = load_data(mnist_train, delimeter, labels)
    test_set = load_data(mnist_test, delimeter, labels)
    
    # decrease dataset size
    train_df = train_df[0 : int(train_df.shape[0]*.1)][:]
    test_set = test_set[0 : int(train_df.shape[0]*.1)][:]

    # split training data into training & validation sets
    split = 0.1
    train_set, valid_set = generate_validation_set(train_df, split)
    
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
    layer_config = [n, int(784/2), 200, 10]
    alpha = 12
    activation_function = "leaky_ReLU"
    const_c = 0.1
    
    # construct neural network
    NN = NeuralNetwork(n, alpha, layer_dimensions=layer_config, activation_func=activation_function, c=const_c)
    
    #train neural network
    epochs = 8
    NN.train(X_train, y_train, X_valid, y_valid, X_test, y_test, epochs)
    NN.evaluate(X_test, y_test)
    print(NN.total_cost(X_test, y_test))
    
    # store or load Neural Network data here
    # ATTENTION - take note of the store and load methods
    NeuralNetworkConfig.store_network_config(NN, 'test_model3')
    #NN = NeuralNetworkConfig.load_network_config('test_model')
    
    # plot cost data
    cost_hist = NN.cost_hist
    plot_cost_hist(cost_hist)

    return

if __name__ == '__main__':
    main()