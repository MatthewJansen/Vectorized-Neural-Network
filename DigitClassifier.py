import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork


def load_data(filename: str, delimeter: str, labels: list):
    return pd.read_csv(filename, sep=delimeter, names=labels)


def generate_validation_set(training_set: pd.DataFrame, split_ratio: float):
    n = training_set.shape[0]

    temp_set = training_set.sample(
        frac=1, random_state=42).reset_index(drop=True)

    new_train_set = temp_set[0: int(n * (1 - split_ratio))][:]
    valid_set = temp_set[int(n * (1 - split_ratio)):][:]

    return new_train_set, valid_set


def one_hot_encoding(digit_set: pd.DataFrame):
    vec_shape = (10, 1)
    def encode(id, vec_shape): return np.array(
        [0 if i != id else 1 for i in range(10)]).reshape(vec_shape)

    encoded_digit_labels = {digit: encode(
        digit, vec_shape) for digit in range(10)}
    encoded_digit_set = digit_set['id'].map(encoded_digit_labels)

    digit_set['id'] = encoded_digit_set

    return digit_set

def evaluate_prediction(y, y_pred):
    return np.argmax(y) == np.argmax(y_pred)


def main():
    mnist_test = 'mnist_test.csv'
    mnist_train = 'mnist_train.csv'

    # get training data
    delimeter = ','
    labels = ['id'] + [f'pixel_{i}' for i in range(784)]

    # get datasets
    train_df = load_data(mnist_train, delimeter, labels) / 255
    test_set = load_data(mnist_test, delimeter, labels) / 255

    # apply one-hot-encoding to data
    train_df = one_hot_encoding(train_df[0 : int(train_df.shape[0]*.1)][:])
    test_set = one_hot_encoding(test_set[0 : int(train_df.shape[0]*.1)][:])

    # split training data into training & validation sets
    split = 0.8
    train_set, valid_set = generate_validation_set(train_df, split)

    #construct input output vector sets
    set_splits = lambda digit_set: (digit_set.iloc[:, 1:].to_numpy(), digit_set.iloc[:, 0].to_numpy())
    X_train, y_train = set_splits(train_set)
    X_valid, y_valid = set_splits(valid_set)
    X_test, y_test = set_splits(test_set)

    # initialise neural network parameters
    n = train_set.shape[1] - 1
    layer_config = [n, 400, 400, 9]
    alpha = 0.01
    activation_function = "ReLU"

    # construct neural network
    NN = NeuralNetwork(n, alpha, layer_config,
                       activation_func=activation_function)

    # X_shape = (784, 1)
    # X1, y1 = X_train[2].reshape(X_shape), y_train[2]
    # y_pred = NN.forward_propagate(X1)
    
    # print(f'y: {np.argmax(y1)}\t\ty_pred: {np.argmax(y_pred)}\n\nCorrect prediction: {evaluate_prediction(y1, y_pred)}') 

    #train neural network
    epochs = 3
    NN.train(X_train, y_train, X_valid, y_valid, epochs)

    return 0


if __name__ == '__main__':
    main()
