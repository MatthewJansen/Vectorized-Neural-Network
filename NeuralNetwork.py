import numpy as np
import time
from ActivationFunctions import ActivationFunctions

class NeuralNetwork():
    '''
    NeuralNetwork
    -------------

    Provides:

    - Generalized neural network data structure
    - Vectorized implementation of: 
        - Forward Propagation
        - Back-Propagation Algorithm
    - Train and Test features


    @params
        - feature_count: (int) -> Number of input features
        - learning_rate: (float) -> Rate at which the Neural Network should learn
        - activation_func: (str) -> Activation function to be used for the Neural Network
                Available activation functions:
                        - Sigmoid
                        - ReLU
                        - leaky_ReLU
        - layer_dimensions: (list) -> A list of integers representing the number of neurons each layer should contain
        - c: (float) -> constant used as a parameter to the ReLU and leaky_ReLU function

    '''

    def __init__(self, feature_count, learning_rate, layer_dimensions=[1, 1], activation_func='Sigmoid', c=0):

        # check if layer_dimensions has at least 2 elements.
        if(len(layer_dimensions) < 2):
            raise(Exception)

        self.feature_count = feature_count
        self.alpha = learning_rate
        self.layer_dimensions = layer_dimensions
        self.activation_func = activation_func
        self.c = 0

        if (activation_func == 'leaky_ReLU') and (c > 0):
            self.c = c

        self.activation, self.activation_deriv = ActivationFunctions.get_activation_pair(activation_func)

        self.neuralnetwork = {}
        self.neuralnetwork['z'] = {}
        self.neuralnetwork['deltas'] = {}

        # set network weights
        # Note: if layer[i] has m neurons and layer[i + 1] has n neurons, then the matrix containing the values of
        #       the weights connecting layer[i, i + 1] has dimmension(n, m).
        self.neuralnetwork['weights'] = {i: np.random.normal(0, 0.09, (layer_dimensions[i], layer_dimensions[i - 1]))
                                         for i in range(1, len(layer_dimensions))}

        # bias should be a column vector with the same amount of entries as nodes per layer
        # (dimmension(n, 1) where n is the number of neurons in a layer)
        self.neuralnetwork['bias'] = {i: np.array([[-1 * np.random.random() for j in range(layer_dimensions[i])]]).T
                                      for i in range(1, len(layer_dimensions))}

        # activation should be a column vector with the same amount of entries as nodes per layer
        # (dimmension(n, 1) where n is the number of neurons in a layer)
        self.neuralnetwork['activation'] = {i: np.array([[0 for j in range(layer_dimensions[i])]]).T
                                            for i in range(len(layer_dimensions))}

        self.cost_hist = {}
        self.accuracies = {}

    def get_cost_hist(self, setname):
        '''
        Returns the cost values during prior neural network training sessions.

        @params
        - setname: (str) -> name of the Neural Network model

        @returns
        - (list) -> list containing cost per epoch from Neural Network training session
        '''
        if setname not in (self.cost_hist).keys:
            raise ValueError(
                f"Neural Network was not evaluated with set named {setname}.")
        return self.cost_hist[setname]

    @staticmethod
    def compute_activation(weights, a_i, bias):
        '''
        Computes the activation of all nodes in the next layer.

        @params
        - weights: (np.ndarray) -> weight matrix for the current layer 
        - a_i: (np.ndarray) -> activation of the current layer
        - bias: (np.ndarray) -> bias vector for the current layer

        @returns
        -
        '''
        activation = np.dot(weights, a_i)
        activation = activation + bias
        return activation

    @staticmethod
    def reshape_vector(x):
        '''
        Reshapes a column vector to insure it has dimension (n, 1)
        given that it has n entries.

        @params
        - x: (np.ndarray) -> the column vector to be reshaped

        @returns
        - (np.ndarray) -> reshaped column vector with shape/dim (n, 1)
        '''
        return x.reshape((np.shape(x)[0], 1))

##################Forward-propagation and Backpropagation############################

    def forward_propagate(self, input_layer):
        '''
        Propagate the input_layer to the output layer of the neural network.

        @params
        - input_layer: (np.ndarray) -> input to the neural network

        @returns
        - output_layer: (np.ndarray) -> neural network output computed by means of forward propagation
        '''
        W = self.neuralnetwork['weights']
        b = self.neuralnetwork['bias']
        A = self.neuralnetwork['activation']
        z = self.neuralnetwork['z']
        A[0] = NeuralNetwork.reshape_vector(input_layer)

        layers = list(W.keys())
        final_layer = layers[-1]
        next_layer = 0

        # Perform feed-forward up to the output layer
        for i in layers[:-1]:
            z[i] = NeuralNetwork.compute_activation(W[i], A[i - 1], b[i])

            if (self.activation_func == 'leaky_ReLU'):
                next_layer = self.activation(self.c, z[i])
            else:
                next_layer = self.activation(z[i])

            A[i] = next_layer

        # Activate the output layer with sigmoid
        z[final_layer] = NeuralNetwork.compute_activation(
            W[final_layer], A[final_layer - 1], b[final_layer])
        # ATTENTION - this can be changed to evaluate with other evaluation functions
        next_layer = ActivationFunctions.sigmoid(z[final_layer])
        A[final_layer] = next_layer

        # store activations of the network
        self.neuralnetwork['activation'] = A
        self.neuralnetwork['z'] = z

        output_layer = A[max(A.keys())]

        return output_layer

    @staticmethod
    def encode_output(y):
        '''
        Encodes the expected output from the dataset into a sparse column vector
        containing the value 1 at the index assigned to the output.

        @param
        - y: (Any) -> expected output from dataset 

        @returns
        - encoded_y: (np.ndarray) -> encoded version of the expected output y

        ATTENTION - this function should be custom written to accommodate the 
        evaluation for the Neural Network output. 
        '''
        encoded_y = np.zeros(((10, 1)))
        encoded_y[int(y)] = 1

        return encoded_y

    def total_cost(self, X: np.ndarray, y: np.ndarray):
        '''
        Computes and returns the average cost to compute outputs for all samples 
        contained in a given dataset.

        @params
        - X: (np.ndarray) -> Input dataset 
        - y: (np.ndarray) -> Target output dataset

        @returns
        - (float) -> Average cost of a Neural Network relative to the provided dataset
        '''
        # initialise useful variables
        cost_C = 0
        n = X.shape[0]

        # compute cost for each input
        for i in range(n):
            # feed input data to the Network
            x_i = NeuralNetwork.reshape_vector(X[i])
            y_i = NeuralNetwork.encode_output(y[i])
            pred = NeuralNetwork.forward_propagate(self, x_i)
            # compute the cumulative cost
            cost_C += sum(NeuralNetwork.cost(pred, y_i))

        # return average cost
        return (1 / n) * cost_C

    @staticmethod
    def cost(output, y):
        '''
        Computes and returns the cost of the neural network output.

        @params
        - output: Neural network output (commonly referred to as y_pred)
        - y: Expected output

        @returns
        - (np.ndarray) -> Cost of the neural network's output
        '''

        # return vector containing the cost for each neuron in the output layer
        return 0.5 * np.square(y - output)

    def cost_gradient(self, output, y):
        '''
        Used to compute the gradient of the neural network output. 

        @params
        - output: (np.ndarray) -> computed Neural Network output
        - y: (np.ndarray) -> expected output

        @returns
        - grad_C: (np.ndarray) -> gradient vector 
        '''
        grad_C = 0

        # compute gradient for the output of the Neural Network relative to the activation function
        if (self.activation_func == 'leaky_ReLU'):
            grad_C = (1 / self.feature_count) * (y - output) * \
                self.activation_deriv(self.c, output)
        if (self.activation_func != 'leaky_ReLU'):
            grad_C = (1 / self.feature_count) * (y - output) * \
                self.activation_deriv(output)

        return grad_C

    def hidden_layer_cost(self, delta, weights, z):
        '''
        Computes the cost of the output relative to the hidden layers
        within the neural network (i.e. the deltas for the hidden layers).

        @params
        - delta: (np.ndarray) -> delta of next layer
        - weights: (np.ndarray) -> weight matrix of the next layer
        - z: (np.ndarray) -> activation vector of the next layer

        @returns
        - (np.ndarray) -> delta for the current layer 
        '''

        # compute useful temporary variable
        delta_W = np.dot(weights.T, delta)

        # compute delta (hidden layer cost) relative to the activation function used
        if (self.activation_func == 'leaky_ReLU'):
            return delta_W * self.activation_deriv(self.c, z)

        return delta_W * self.activation_deriv(z)

    def back_propagate_error(self, y):
        '''
        Backpropagate the error/cost of the output layer to the input layer
        (computes all the deltas for all layers within the neural network)

        @params
        - y: (np.ndarray) -> expected output for the neural network input

        ATTENTION - y is encoded here to match the dimmensions of the output layer

        @returns
        - None
        '''

        # copy useful network variables
        W = self.neuralnetwork['weights']
        A = self.neuralnetwork['activation']
        z = self.neuralnetwork['z']
        y_encoded = self.encode_output(y)  # ATTENTION!
        deltas = {}

        # Compute delta for the output layer
        output = A[max(A.keys())]
        output_error_grad = NeuralNetwork.cost_gradient(
            self, output, y_encoded)
        deltas[max(A.keys())] = output_error_grad

        # Construct a list containing layer numbers to be traversed
        layers = list(A.keys())[2:]

        # Traverse the network backwards from the output layer up to
        # the second last layer and compute hidden deltas
        for layer in reversed(layers):
            deltas[layer - 1] = NeuralNetwork.hidden_layer_cost(
                self, deltas[layer], W[layer], z[layer - 1])

        # Add deltas to the network structure
        self.neuralnetwork['deltas'] = deltas

        return

    def update_network(self):
        '''
        Updates the weights and biases of the Neural Network using the deltas 
        computed during Backpropagation.

        Update rules:

        - W[i] := W[i] + (alpha * dw[i])
        - b[i] := b[i] + (alpha * db[i])

        ATTENTION - dw[i] is already negative due to how the cost function is 
        defined for the Neural Network. 
        @params
        - None

        @returns
        - None
        '''
        # initialise useful variables
        dw = {}
        db = {}

        W = self.neuralnetwork['weights']
        b = self.neuralnetwork['bias']
        A = self.neuralnetwork['activation']
        z = self.neuralnetwork['z']

        z[0] = A[0]
        deltas = self.neuralnetwork['deltas']

        final_layer = max(A.keys())
        dw[final_layer] = np.dot(deltas[final_layer], A[final_layer - 1].T)
        # np.sum(deltas[final_layer], axis=1, keepdims=True)
        db[final_layer] = (1 / self.feature_count) * deltas[final_layer]

        # compute differentials for weights and biases of each layer in the network
        # starting at the output layer
        for layer in reversed(list(W.keys())[1:]):
            # Compute derivative of activation function
            func_deriv = 0
            if (self.activation_func == 'leaky_ReLU'):
                func_deriv = self.activation_deriv(self.c, z[layer - 1])
            if (self.activation_func != 'leaky_ReLU'):
                func_deriv = self.activation_deriv(z[layer - 1])

            dw[layer - 1] = (1 / self.feature_count) * \
                np.dot(W[layer].T, deltas[layer]) * func_deriv
            db[layer - 1] = (1 / self.feature_count) * deltas[layer - 1]

        # compute new weights and biases
        for i in W.keys():
            W[i] += self.alpha * dw[i]
            b[i] += self.alpha * db[i]

        # update weights and biases
        self.neuralnetwork['weights'] = W
        self.neuralnetwork['bias'] = b

        return

    def evaluate(self, X, y):
        '''
        Evaluates the accuracy of the Neural Network relative to the data
        provided.

        @params
        - X: (np.ndarray) -> input data from the dataset
        - y: (np.ndarray) -> output data from the dataset

        @returns
        - accuracy: (float) -> accuracy of the neural network relative to the provided data
        '''
        # get dataset size
        dataset_size = np.size(X, 0)

        # initialise variables used to track no. of entries classified correctly
        positive = 0
        negative = 0

        # compute accuracies for each entry
        for i in range(dataset_size):
            # forward-propagate input
            x = X[i][:]
            prediction = NeuralNetwork.forward_propagate(self, x)

            # compare network output to expected output
            evaluation = 1 if (y[i] == np.argmax(prediction)) else 0

            # track comparison
            positive += 1 if (evaluation == 1) else 0
            negative += 1 if (evaluation == 0) else 0

        # compute total accuracy
        accuracy = (positive / dataset_size) * 100

        return accuracy

    def train(self, X_train, y_train, X_valid, y_valid, X_test, y_test, epochs):
        '''
        Train the Neural Network for a number of epochs given data from 
        the training dataset, validation dataset and testing dataset.

        @params
        - X_train: (np.ndarray) -> Training input data
        - y_train: (np.ndarray) -> Training target output data
        - X_valid: (np.ndarray) -> Validation input data
        - y_valid: (np.ndarray) -> Validation target output data
        - X_test: (np.ndarray) -> Testing input data 
        - y_test: (np.ndarray) -> Testing target output data
        - epochs: (int) -> number of iterations data gets passed through the Neural Network 

        @returns
        - None
        '''
        initial_feedback = f'Data pre-processing complete, training initiated...\n' \
            + f'================================================='
        print(initial_feedback)
        # initialise useful variables
        epoch = 1
        train_size = np.size(X_train, 0)

        Train_accuracies = []
        Valid_accuracies = []

        train_cost_hist = []
        valid_cost_hist = []

        total_training_time = 0

        def format_time(time):
                s, ms = divmod(time, 1)
                m, s = divmod(s / 60, 1)
                s = round(s * 60)
                h, m = divmod(m / 60, 1)
                m = round(m * 60)

                return f'{int(h)}h {int(m)}m {int(s)}.{str(round(ms, 4))[2:]}s'

        # train neural network
        while (epoch <= epochs):
            # record training time
            epoch_start = time.time()

            # loop through
            for i in range(train_size):
                # reshape input vector
                x = NeuralNetwork.reshape_vector(X_train[i][:])

                # feed input, train & update neural network
                NeuralNetwork.forward_propagate(self, x)
                NeuralNetwork.back_propagate_error(self, y_train[i])
                NeuralNetwork.update_network(self)

            # compute epoch execution time and update total training time record
            epoch_time = time.time() - epoch_start
            total_training_time += epoch_time

            # compute training & valdation accuracy
            train_accuracy = NeuralNetwork.evaluate(self, X_train, y_train)
            valid_accuracy = NeuralNetwork.evaluate(self, X_valid, y_valid)

            # compute training and test loss
            train_epoch_cost = float(
                NeuralNetwork.total_cost(self, X_train, y_train))
            valid_epoch_cost = float(
                NeuralNetwork.total_cost(self, X_valid, y_valid))

            # print epoch stats
            feedback = f"Epoch: [{epoch}]\t\tExecution time: [{format_time(epoch_time)}]\n\n" \
                f'Train cost: {train_epoch_cost}\t\t\tValidation cost: {valid_epoch_cost}\n' \
                f'Train accuracy: {round(train_accuracy, 4)}%\t\t\tValidation accuracy: {round(valid_accuracy, 4)}%\n' \
                '-------------------------------------------------'
            print(feedback)

            # store useful data in lists
            Train_accuracies.append(train_accuracy)
            Valid_accuracies.append(valid_accuracy)

            train_cost_hist.append(train_epoch_cost)
            valid_cost_hist.append(valid_epoch_cost)

            # update epoch
            epoch += 1

        test_cost = NeuralNetwork.total_cost(self, X_test, y_test)
        test_accuracy = NeuralNetwork.evaluate(self, X_test, y_test)
        
        final_feedback = f'Neural Network performance on test set:\n' \
            f'Test cost: {test_cost}\t\t\tTest accuracy: {round(test_accuracy, 4)}%\n' \
            f'Total training time: {format_time(total_training_time)}\n' \
            f'================================================='
        print(final_feedback)

        # set useful data for storage purposes
        self.accuracies['Train_set'] = Train_accuracies
        self.accuracies['Validation_set'] = Valid_accuracies
        self.cost_hist['Train_cost'] = train_cost_hist
        self.cost_hist['Validation_cost'] = valid_cost_hist

        return
