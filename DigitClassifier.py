import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
from NetworkConfigHandler import NeuralNetworkConfig
import matplotlib.pyplot as plt


def load_data(filename: str, delimeter: str, labels: list):
    return pd.read_csv(filename, sep=delimeter, names=labels)


def generate_validation_set(training_set: pd.DataFrame, split_ratio: float):
    n = training_set.shape[0]

    temp_set = training_set.sample(frac=1, random_state=42).reset_index(drop=True)

    new_train_set = temp_set[0: int(n * (1 - split_ratio))][:]
    valid_set = temp_set[int(n * (1 - split_ratio)):][:]

    return new_train_set, valid_set

def encode_output(y):
    """
    Encodes the expected output from the dataset into a sparse column vector
    containing the value 1 at the index assigned to the output.

    @param
    y:  output from dataset 
    
    Note - this function should be written to accommodate the evaluation for 
    the Neural Network. 
    """
    encoded_vec = np.zeros((10, 1))
    encoded_vec[y] = 1
    return encoded_vec

def plot_cost_hist(cost_history: list):
    epochs = [i+1 for i in range(len(cost_history))]
    plt.plot(epochs, cost_history)
    plt.xlabel('epochs')
    plt.ylabel('Cost J')
    plt.title('Cost history plot')
    plt.show()
    return

def main():
    mnist_test = 'mnist_test.csv' #'/content/drive/MyDrive/MachineLearning/DigitRecognition/mnist_test.csv'
    mnist_train = 'mnist_train.csv' #'/content/drive/MyDrive/MachineLearning/DigitRecognition/mnist_train.csv'

    # get datasets
    delimeter = ','
    labels = ['id'] + [f'pixel_{i}' for i in range(784)]

    train_df = load_data(mnist_train, delimeter, labels)
    test_set = load_data(mnist_test, delimeter, labels)
    
    # decrease dataset size
    # train_df = train_df[0 : int(train_df.shape[0]*.1)][:]
    # test_set = test_set[0 : int(train_df.shape[0]*.1)][:]

    # split training data into training & validation sets
    split = 0.1
    train_set, valid_set = generate_validation_set(train_df, split)
    print(train_set.shape, valid_set.shape)
    
    #construct input output vector sets
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
    layer_config = [n, int(784/8), 16, 10]
    alpha = 12
    activation_function = "leaky_ReLU"
    const_c = 0.1
    
    # construct neural network
    NN = NeuralNetwork(n, alpha, layer_config, activation_func=activation_function, c=const_c)
    

    #train neural network
    epochs = 3
    Train_accuracies, Valid_accuracies, cost_hist = NN.train(X_train, y_train, X_valid, y_valid, X_test, y_test, epochs)
    NN.evaluate(X_test, y_test)
    print(NN.total_cost(X_test, y_test))
    
    NeuralNetworkConfig.store_network_config(NN, 'test_model')

    return Train_accuracies, Valid_accuracies, cost_hist

if __name__ == '__main__':
    Train_accuracies, Valid_accuracies, cost_hist = main()
    plot_cost_hist(cost_hist)
