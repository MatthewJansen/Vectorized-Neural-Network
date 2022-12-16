import numpy as np
from tqdm import tqdm
import time
from ActivationFunctions import ActivationFunctions
from DataHandler import DataHandler

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
                        - sigmoid
                        - ReLU
                        - leaky_ReLU
                        - tanh
        - layer_dimensions: (list) -> A list of integers representing the number of neurons each layer should contain
        - c: (float) -> constant used as a parameter to the ReLU and leaky_ReLU function

    '''

    def __init__(self, config: list, learning_rate):

        # check if layer_dimensions has at least 2 elements.
        # if(len(layer_dimensions) < 2):
        #     raise(Exception)

        self.config = config
        self.feature_count = self.config[0]['units']
        self.layer_dimensions = [self.config[_]['units'] for _ in range(len(self.config))]
        self.layer_activations = {_: config[_]['activation'] for _ in range(len(config))}
        self.alpha = learning_rate

        self.activation_functions = {
            _: ActivationFunctions.get_activation_pair(config[_]['activation'])
                for _ in range(len(self.config))
        }

        self.neuralnetwork = {}
        self.neuralnetwork['z'] = {}
        self.neuralnetwork['deltas'] = {}

        # set network weights
        # Note: if layer[i] has m neurons and layer[i + 1] has n neurons, 
        #       then the matrix containing the values of the weights 
        #       connecting layer[i, i + 1] has dimension(n, m).
        mu = 0
        sigma = 1 / (self.layer_dimensions[0] ** 0.5)
        self.generate_weights(mu, sigma)

        # bias should be a column vector with the same amount of entries 
        # as nodes per layer (dimension(n, 1) where n is the number of 
        # neurons in a layer)
        self.generate_bias() 
        
        # activation should be a column vector with the same amount of 
        # entries as nodes per layer (dimension(n, 1) where n is the 
        # number of neurons in a layer)
        self.neuralnetwork['activation'] = {i: np.zeros((self.layer_dimensions[i], 1))
                                            for i in range(len(self.layer_dimensions))}

        self.cost_hist = {}
        self.accuracies = {}


    def __call__(self, x:np.ndarray):
        return self.forward_propagate(x.reshape(-1, 1))

    def generate_weights(self, mu, sigma):
        network_weights = {}

        for i in range(1, len(self.layer_dimensions)):
            prev_layer = self.layer_dimensions[i - 1]
            current_layer = self.layer_dimensions[i]
            network_weights[i] = np.random.normal(mu, sigma, (current_layer, prev_layer))

        self.neuralnetwork['weights'] = network_weights

        return

    def generate_bias(self):
        network_bias = {}

        for i in range(1, len(self.layer_dimensions)):
            current_layer = self.layer_dimensions[i]
            network_bias[i] = -1 * np.random.rand(current_layer, 1)
            
        self.neuralnetwork['bias'] = network_bias

        return


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

############################
# Forward-propagation and Backpropagation
############################

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
        A[0] = input_layer

        layers = list(W.keys())
        # final_layer = layers[-1]
        next_layer = 0

        # Perform feed-forward up to the output layer
        for i in layers:
            z[i] = NeuralNetwork.compute_activation(W[i], A[i - 1], b[i])
            layer_activation = self.activation_functions[i][0]
            next_layer = layer_activation(z[i])
            # if (self.activation_func == 'leaky_ReLU'):
            #     next_layer = self.activation(self.c, z[i])
            # else:
            #     next_layer = self.activation(z[i])

            A[i] = next_layer

        # # Activate the output layer with sigmoid
        # z[final_layer] = NeuralNetwork.compute_activation(
        #     W[final_layer], A[final_layer - 1], b[final_layer])
        # # ATTENTION - this can be changed to evaluate with other evaluation functions
        # next_layer = self.activation_deriv(self.c, z[final_layer])
        # A[final_layer] = next_layer

        # store activations of the network
        self.neuralnetwork['activation'] = A
        self.neuralnetwork['z'] = z

        output_layer = A[max(A.keys())]

        return output_layer


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
        n = len(X)

        # compute cost for each input
        for i in range(n):
            # feed input data to the Network
            x = X[i].reshape(-1, 1)
            pred = NeuralNetwork.forward_propagate(self, x)

            # compute the cumulative cost
            cost_C += sum(NeuralNetwork.cost(pred, y[i].reshape(-1, 1)))

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

    def cost_gradient(self, y_true, y_pred, activation_deriv):
        '''
        Used to compute the gradient of the neural network output. 

        @params
        - output: (np.ndarray) -> computed Neural Network output
        - y: (np.ndarray) -> expected output

        @returns
        - grad_C: (np.ndarray) -> gradient vector 
        '''
        cost_deriv = (1 / self.feature_count) * (y_true - y_pred)

        # compute gradient for the output of the Neural Network relative to the activation function
        grad_C = cost_deriv * activation_deriv(y_pred)

        # if (self.activation_func == 'leaky_ReLU'):
        #     grad_C = cost_deriv * self.activation_deriv(self.c, y_pred)
        
        # if (self.activation_func != 'leaky_ReLU'):
        #     grad_C = cost_deriv * (y_true - y_pred) * self.activation_deriv(y_pred)

        return grad_C

    def hidden_layer_cost(self, delta, weights, z, activation_deriv):
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
        # if (self.activation_func == 'leaky_ReLU'):
        #     return delta_W * self.activation_deriv(self.c, z)

        return delta_W * activation_deriv(z) #self.activation_deriv(z)

    def back_propagate_error(self, y_true):
        '''
        Backpropagate the error/cost of the output layer to the input layer
        (computes all the deltas for all layers within the neural network)

        @params
        - y_true: (np.ndarray) -> expected output for the neural network input

        @returns
        - None
        '''

        # copy useful network variables
        W = self.neuralnetwork['weights']
        A = self.neuralnetwork['activation']
        z = self.neuralnetwork['z']
        deltas = {}

        # Compute delta for the output layer
        output_layer_idx = max(A.keys())
        y_pred = A[output_layer_idx]
        output_error_grad = self.cost_gradient(
            y_true.reshape(-1, 1), 
            y_pred, 
            self.activation_functions[output_layer_idx][1] # final layer derivative
        )
        deltas[output_layer_idx] = output_error_grad

        # Construct a list containing layer numbers to be traversed
        layers = list(A.keys())[2:]

        # Traverse the network backwards from the output layer up to
        # the second last layer and compute hidden deltas
        for layer in reversed(layers):
            deltas[layer - 1] = NeuralNetwork.hidden_layer_cost(
                self, deltas[layer], 
                W[layer], z[layer - 1], 
                self.activation_functions[layer][1]
            )

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
        N_inv = (1 / self.feature_count)

        final_layer = max(A.keys())
        dw[final_layer] = np.dot(deltas[final_layer], A[final_layer - 1].T)
        # np.sum(deltas[final_layer], axis=1, keepdims=True)
        db[final_layer] = N_inv * deltas[final_layer]

        # compute differentials for weights and biases of each layer in the network
        # starting at the output layer
        for layer in reversed(list(W.keys())[1:]):
            # Compute derivative of activation function
            func_deriv = self.activation_functions[layer][1](z[layer - 1])
            
            # if (self.activation_func == 'leaky_ReLU'):
            #     func_deriv = self.activation_deriv(self.c, z[layer - 1])
            # if (self.activation_func != 'leaky_ReLU'):
            #     func_deriv = self.activation_deriv(z[layer - 1])

            dw[layer - 1] = N_inv * np.dot(W[layer].T, deltas[layer]) * func_deriv
            db[layer - 1] = N_inv * deltas[layer - 1]

        # compute new weights and biases
        for i in W.keys():
            W[i] += self.alpha * dw[i]
            b[i] += self.alpha * db[i]

        # update weights and biases
        self.neuralnetwork['weights'] = W
        self.neuralnetwork['bias'] = b

        return

    def get_predictions(self, ds: DataHandler):
        '''
        Computes predictions vector consisting of all predictions for each sample in X and 
        has the same shape as y.

        @params
        - X: (np.ndarray) -> input data from the dataset
        - y: (np.ndarray) -> output data from the dataset

        @returns
        - y_pred: (np.ndarray) -> predictions for all samples in X with the same shape as y
        '''

        y_pred = np.zeros_like(ds.y)

        for i in range(len(ds)):
            x, target = ds[i]
            y_pred[i] = np.argmax(self.forward_propagate(x.reshape(-1, 1)))
        
        return y_pred

    def evaluate(self, ds: DataHandler):
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
        ds_size = len(ds)
        ds_size_inv = 1 / ds_size
        
        # initialise variables used to track no. of entries classified correctly
        accuracy = 0
        cost = 0

        # compute accuracies for each entry
        for i in range(ds_size):
            x, target = ds[i]

            # forward-propagate input
            x = x.reshape(-1, 1)
            pred = NeuralNetwork.forward_propagate(self, x)

            # compare network output to expected output and track accuracy
            comparison = 1 if (np.argmax(target.reshape(-1, 1)) == np.argmax(pred)) else 0
            
            accuracy += ds_size_inv * (comparison)
            cost += ds_size_inv * np.sum(NeuralNetwork.cost(pred, target))

        return cost, accuracy

    def fit(self, train_ds, val_ds, epochs):
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
        initial_feedback = f'Data received -> Training initiated...\n' \
            + f'================================================='
        print(initial_feedback)

        # initialise variables
        epoch = 1
        train_size = len(train_ds)
        train_size_inv = 1 / train_size

        Train_accuracies = []
        Valid_accuracies = []

        train_cost_hist = []
        valid_cost_hist = []

        total_training_time = 0

        compare = lambda pred, truth: 1 if (np.argmax(truth) == np.argmax(pred)) else 0

        # def format_time(time):
        #         s, ms = divmod(time, 1)
        #         m, s = divmod(s / 60, 1)
        #         s = round(s * 60)
        #         h, m = divmod(m / 60, 1)
        #         m = round(m * 60)

        #         return f'{int(h)}h {int(m)}m {int(s)}.{str(round(ms, 4))[2:]}s'

        # train neural network
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')

            train_accuracy = 0
            train_epoch_cost = 0

            # record training time
            epoch_start = time.time()

            # loop through
            for i in tqdm(range(train_size)):

                inputs, target = train_ds[i]
                # reshape input vector
                x = NeuralNetwork.reshape_vector(inputs)

                # feed input, train & update neural network
                pred = NeuralNetwork.forward_propagate(self, x)
                
                # log cost and accuracy
                train_accuracy += train_size_inv * (compare(pred, target))
                train_epoch_cost += train_size_inv * np.sum(NeuralNetwork.cost(pred, target))
                
                # backpropagate and update
                NeuralNetwork.back_propagate_error(self, target)
                NeuralNetwork.update_network(self)



            # compute epoch execution time and update total training time record
            epoch_time = time.time() - epoch_start
            total_training_time += epoch_time

            # compute training & valdation accuracy
            valid_epoch_cost, valid_accuracy = NeuralNetwork.evaluate(self, val_ds)

            # compute training and test loss
            # valid_epoch_cost = float(
            #     NeuralNetwork.total_cost(self, val_ds.X, val_ds.y))

            # print epoch stats
            # feedback = f"Epoch: [{epoch}]\t\tExecution time: [{format_time(epoch_time)}]\n\n" \
            #     f'Train cost: {train_epoch_cost}\t\t\tValidation cost: {valid_epoch_cost}\n' \
            #     f'Train accuracy: {round(train_accuracy* 100, 4)}%\t\t\tValidation accuracy: {round(valid_accuracy* 100, 4)}%\n' \
            #     '-------------------------------------------------'

            print(f'cost: {round(train_epoch_cost, 6)}\t\t\tval_cost: {round(valid_epoch_cost, 6)}')
            print(f'accuracy: {round(train_accuracy, 6)}\t\tval_accuracy: {round(valid_accuracy, 6)}' )
            print('-------------------------------------------------')

            # store useful data in lists
            Train_accuracies.append(train_accuracy)
            Valid_accuracies.append(valid_accuracy)

            train_cost_hist.append(train_epoch_cost)
            valid_cost_hist.append(valid_epoch_cost)

            # # update epoch
            # epoch += 1

        # test_cost = NeuralNetwork.total_cost(self, train_ds.X, train_ds.y)
        # test_accuracy = NeuralNetwork.evaluate(self, val_ds.X, val_ds.y)
        
        # final_feedback = f'Neural Network performance on test set:\n' \
        #     f'Test cost: {test_cost}\t\t\tTest accuracy: {round(test_accuracy * 100, 4)}%\n' \
        #     f'Total training time: {format_time(total_training_time)}\n' \
        #     f'================================================='
        # print(final_feedback)

        # set useful data for storage purposes
        self.accuracies['Train_set'] = Train_accuracies
        self.accuracies['Validation_set'] = Valid_accuracies
        self.cost_hist['Train_cost'] = train_cost_hist
        self.cost_hist['Validation_cost'] = valid_cost_hist

        return
