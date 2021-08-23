from NeuralNetwork import NeuralNetwork
import numpy as np

def random_one_zero() -> int:
    return 0 if np.random.random() < 0.5 else 1

def generate_sample():
    s = [float(random_one_zero()), float(random_one_zero())]
    return s + [float(s[0] == s[1])]

def generate_data(sample_size: int):
    return np.array([generate_sample() for i in range(sample_size)])

def prepare_data(sample_size, data):
    X, y = data[:,:-1], np.reshape(data[:, -1], (sample_size, 1))
    return X, y

def get_sample(index, X, y):
    return np.reshape(X[index], (X[index].shape[0], 1)), y[index]

def split_dataset(data, sample_size, ratio):
    set1 = data[:int(sample_size * ratio), :]
    set2 = data[int(sample_size * ratio):, :]
    return set1, set2

def evaluate(y, pred) -> bool:
    y_bar = float(pred >= 0.5)
    return bool(y == y_bar)

def main():
    n = 2
    layer_config = [n, 2, 2, 1]
    alpha = 0.1
    activation_function = "ReLU"
    NN = NeuralNetwork(n, alpha, layer_config, activation_func= activation_function)

    sample_size = 1000
    data = generate_data(sample_size)
    
    train_set, test_set = split_dataset(data, sample_size, 0.8) 
    train_set, validation_set = split_dataset(train_set, np.shape(train_set)[0], 0.75)
    
    X_train, y_train = prepare_data(np.shape(train_set)[0], train_set)
    X_valid, y_valid = prepare_data(np.shape(validation_set)[0], validation_set)
    X_test, y_test = prepare_data(np.shape(test_set)[0], test_set)

    x, y = get_sample(1, X_train, y_train)

    NN.train(X_train, y_train, X_valid, y_valid)

    pred = NN.forward_propagate(x)

    print(f"Input:\n{x}\n\nExpected Output:\n{y}\n\nPrediction:\n{float(pred >= 0.5)}\n\nCorrect: {evaluate(actual_y, pred)}")
    return

if __name__ == "__main__":
    #print(generate_data(20))
    main()