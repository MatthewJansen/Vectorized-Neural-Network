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

def evaluate(y, pred) -> bool:
    y_bar = float(pred >= 0.5)
    return bool(y == y_bar)

def main():
    n = 2
    layer_config = [n, 2, 2, 1]
    alpha = 0.1
    activation_function = "ReLU"
    NN = NeuralNetwork(n, alpha, layer_config, activation_func= activation_function)

    sample_size = 20
    data = generate_data(sample_size)
    X, y = prepare_data(sample_size, data)
    x, actual_y = get_sample(1, X, y)

    pred = NN.forward_propagate(x)

    print(f"Input:\n{x}\n\nExpected Output:\n{actual_y}\n\nPrediction:\n{float(pred >= 0.5)}\n\nCorrect: {evaluate(actual_y, pred)}")
    return

if __name__ == "__main__":
    #print(generate_data(20))
    main()