from NeuralNetwork import NeuralNetwork
import numpy as np

def random_one_zero() -> int:
    return 0 if np.random.random() < 0.5 else 1

def generate_sample(n):
    s = [float(random_one_zero()) for i in range(n)]
    return s + [float(s.count(0) == n or s.count(1) == n)]

def generate_data(sample_size: int, n: int):
    return np.array([generate_sample(n) for i in range(sample_size)])

def prepare_data(sample_size, data):
    X, y = data[:,:-1], np.reshape(data[:, -1], (sample_size, 1))
    return X, y

def get_sample(index, X, y):
    return np.reshape(X[index], (X[index].shape[0], 1)), y[index]

def split_dataset(data, sample_size, ratio):
    set1 = data[:int(sample_size * ratio), :]
    set2 = data[int(sample_size * ratio):, :]
    return set1, set2

def one_hot_encoding(dataset: np.ndarray):
    return np.where(dataset == 1, np.array([[0, 1]]), np.array([[1, 0]]))

def evaluate(y, pred) -> bool:
    y_bar = np.argmax(pred)
    return np.argmax(y) == y_bar

####################################

def main():
    n = 3
    layer_config = [n, 9, 6, 2]
    alpha = 0.003

    activation_function = "tanh"
    NN = NeuralNetwork(n, alpha, layer_config, activation_func= activation_function)

    sample_size = 1000
    data = generate_data(sample_size, n)
    
    train_set, test_set = split_dataset(data, sample_size, 0.9) 
    train_set, validation_set = split_dataset(train_set, np.shape(train_set)[0], 0.75)
    
    X_train, y_train = prepare_data(np.shape(train_set)[0], train_set)
    X_valid, y_valid = prepare_data(np.shape(validation_set)[0], validation_set)
    X_test, y_test = prepare_data(np.shape(test_set)[0], test_set)

    y_train = one_hot_encoding(y_train).reshape(y_train.shape[0], 2, 1)
    y_valid = one_hot_encoding(y_valid).reshape(y_valid.shape[0], 2, 1)
    y_test = one_hot_encoding(y_test).reshape(y_test.shape[0], 2, 1)

    x, y = get_sample(1, X_test, y_test)

    NN.train(X_train, y_train, X_valid, y_valid, 10)
    pred = NN.forward_propagate(x)


    print(f"Input:\n{x}\n\nExpected Output:\n{np.argmax(y)}\n\nPrediction:\n{np.argmax(pred)}\n\nCorrect: {evaluate(y, pred)}")
    
    return

if __name__ == "__main__":
    #print(generate_data(20))
    main()