import numpy as np

class NeuralNetwork:
    """
    NeuralNetwork
    -------------
    Provides:

    - Generalized neural network data structure
    - Vectorized implementation of: 
        - Forward Propagation
        - Back-Propagation Algorithm
    - Train and Test features
    """

    def __init__(self, feature_count, learning_rate, layer_dimensions=[1, 1], activation_func='Sigmoid', c=0):
        """
        Neural network constructor.

        Arguments:
        - feature_count (int): Number of input features
        - learning_rate (float): Rate at which the Neural Network should learn
        - activation (function): Activation function to be used for the Neural Network
                Available activation functions:
                        - Sigmoid
                        - ReLU
                        - leaky_ReLU
        - layer_dimensions (list): A list of integers representing the number of neurons each layer should contain

        Returns:
        - None
        """
        # check if layer_dimensions has at least 2 elements.
        if(len(layer_dimensions) < 2):
            raise(Exception)
        
        self.feature_count = feature_count
        self.alpha = learning_rate

        self.activation_func = activation_func
        self.c = 0
        if activation_func == 'ReLu' and c == 0:
            pass

        if (activation_func == 'leaky_ReLU') and (c > 0):
            self.c = c

        self.activation, self.activation_deriv = NeuralNetwork.set_activation_func(activation_func)

        self.neuralnetwork = {}
        self.neuralnetwork['z'] = {}
        self.neuralnetwork['deltas'] = {}
        
        # set network weights
        # Note: if layer[i] has m neurons and layer[i + 1] has n neurons, then the matrix containing the values of
        #       the weights connecting layer[i, i + 1] has dimmension(n, m).
        self.neuralnetwork['weights'] = {i: np.random.normal(0, 0.9, (layer_dimensions[i], layer_dimensions[i - 1]))
                                          for i in range(1, len(layer_dimensions))}

        # bias should be a column vector with the same amount of entries as nodes per layer
        # (dimmension(n, 1) where n is the number of neurons in a layer)
        self.neuralnetwork['bias'] = {i: np.array([[-1 * np.random.random() for j in range(layer_dimensions[i])]]).T
                                       for i in range(1, len(layer_dimensions))}

        # activation should be a column vector with the same amount of entries as nodes per layer
        # (dimmension(n, 1) where n is the number of neurons in a layer)
        self.neuralnetwork['activation'] = {i: np.array([[0 for j in range(layer_dimensions[i])]]).T
                                             for i in range(len(layer_dimensions))}
    @staticmethod
    def set_activation_func(activation_func):

        func_set = {
            'Sigmoid': [NeuralNetwork.sigmoid, NeuralNetwork.sigmoid_deriv],
            'tanh': [NeuralNetwork.tanh, NeuralNetwork.tanh_deriv],
            'ReLU': [NeuralNetwork.ReLU, NeuralNetwork.ReLU_deriv],
            'leaky_ReLU': [NeuralNetwork.leaky_ReLU, NeuralNetwork.leaky_ReLU_deriv]
        }

        return func_set[activation_func][0], func_set[activation_func][1]

    @staticmethod
    def compute_activation(weights, inputs, bias):
        """Computes the activation of all nodes in the next layer."""
        activation = np.dot(weights, inputs)
        activation = activation + bias
        return activation

##################Activation Functions############################

    @staticmethod
    def sigmoid(z):
        """Compute the sigmoid function for a given input z."""
        return (1 / (1 + np.exp(-1*z)))

    @staticmethod
    def sigmoid_deriv(z):
        """Compute the derivative of the sigmoid function for a given input z."""
        sig = NeuralNetwork.sigmoid(z)
        return (sig * (1 - sig))

    @staticmethod
    def ReLU(z):
        return np.where(z > 0, z, 0)

    @staticmethod
    def ReLU_deriv(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_ReLU(c, z):
        return np.where(z > 0, z, c * z)        
    
    @staticmethod
    def leaky_ReLU_deriv(c, z):
        return np.where(z > 0, 1, c)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_deriv(z):
        return 1 - (NeuralNetwork.tanh(z) ** 2)

#####################################################

    def forward_propagate(self, input_layer):
        """Propagate the input_layer to the output layer of the neural network."""
        W = self.neuralnetwork['weights']
        b = self.neuralnetwork['bias']
        A = self.neuralnetwork['activation']
        z = self.neuralnetwork['z']
        A[0] = input_layer

        layers = list(W.keys())
        final_layer = list(W.keys())[-1]
        next_layer = 0
        
        # Perform feed-forward up to the output layer
        for i in layers[:-1]:
            z[i] = NeuralNetwork.compute_activation(W[i], A[i - 1], b[i])
            
            if (self.activation_func in ['ReLU', 'leaky_ReLU']):
                next_layer = self.activation(self.c, z[i])
            else:
                next_layer = self.activation(z[i])
            
            A[i] = next_layer

        # Activate the output layer with sigmoid
        z[final_layer] = NeuralNetwork.compute_activation(W[final_layer], A[final_layer - 1], b[final_layer])
        next_layer = NeuralNetwork.ReLU(z[final_layer])#.sigmoid(z[final_layer])
        A[final_layer] = next_layer

        # store activations of the network
        self.neuralnetwork['activation'] = A
        self.neuralnetwork['z'] = z

        output_layer = A[max(A.keys())]

        return output_layer

    @staticmethod
    def create_target_vector(target_output):
        """
        Creates a column vector of zeros and 1 as the entry corresponding to the
        expected target output.
        """
        target_vec = np.zeros((10, 1))
        target_vec[target_output] = 1

        return target_vec

    @staticmethod
    def total_output_error(output_layer, target_output):
        """
        Computes the total error in the output of the neural network by
        means of using the SSE error function.
        """
        return 0.5 * np.sum(np.square(output_layer - target_output))

    @staticmethod
    def output_error_vector(output_layer, target_output):
        """
        Constructs a column vector containing the output error of each 
        neuron in the output layer.
        """
        return 0.5 * np.square(output_layer - target_output)
        
    def output_error_gradient(self, output_layer, target_output):
        """Computes the gradient of the output error vector."""
        comp = 0

        if (self.activation_func in ['ReLU', 'leaky_ReLU']):
            deriv_comp = self.activation_deriv(self.c, output_layer)
        else:
            deriv_comp = self.activation_deriv(output_layer)

        return (1 / self.feature_count) * (output_layer - target_output) * deriv_comp

    def hidden_error(self, delta, weights, z):
        """Computes the error gradient vector for a hidden layer."""
        delta_W = np.dot(weights.T, delta)
        func_deriv = 0

        if (self.activation_func in ['ReLU', 'leaky_ReLU']):
            func_deriv = self.activation_deriv(self.c, z)
        else:
            func_deriv = self.activation_deriv(z)

        #func_deriv = NeuralNetwork.leaky_ReLU_deriv(0.01, z)

        return delta_W * func_deriv

    def back_propagate_error(self, target_output):
        """Backpropagate the error of the output layer to the input layer."""
        W = self.neuralnetwork['weights']
        b = self.neuralnetwork['bias']
        A = self.neuralnetwork['activation']
        z = self.neuralnetwork['z']
        deltas = {}

        #Compute delta for the output layer
        output_layer = A[max(A.keys())]
        output_error_grad = NeuralNetwork.output_error_gradient(self, output_layer, target_output)
        deltas[max(A.keys())] = output_error_grad
        
        #Construct a list containing layer numbers to be traversed
        layers = list(A.keys())[2:]

        #Traverse the network backwards from the output layer up to
        #the second last layer and compute hidden deltas
        for layer in reversed(layers):
            deltas[layer - 1] = NeuralNetwork.hidden_error(self, deltas[layer], W[layer], z[layer - 1])

        #Add deltas to the network structure
        self.neuralnetwork['deltas'] = deltas

        return

    def update_network(self):
        """Updates the weights and biases of the Neural Network."""
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
        db[final_layer] = (1 / self.feature_count) * np.sum(deltas[final_layer], axis=1, keepdims=True)

        for layer in reversed(list(W.keys())[1:]):
            # Compute derivative of activation function
            func_deriv = 0
            if (self.activation_func in ['ReLU', 'leaky_ReLU']):
                func_deriv = self.activation_deriv(self.c, z[layer - 1])
            else:
                func_deriv = self.activation_deriv(z[layer - 1])
            
            dw[layer - 1] = (1 / self.feature_count) * np.dot(W[layer].T, deltas[layer]) * func_deriv
            db[layer - 1] = (1 / self.feature_count) * np.sum(deltas[layer - 1], axis=1, keepdims=True)

        for i in W.keys():
            W[i] += -1 * self.alpha * dw[i]
            b[i] += self.alpha * db[i] 

        self.neuralnetwork['weights'] = W
        self.neuralnetwork['bias'] = b

        return

    def evaluate(self, test_input, test_target):
        """Test the Neural Network."""
        test_size = np.size(test_input, 0)
        positive = 0
        negative = 0

        for i in range(test_size):
            x = test_input[i]#[:].reshape((np.shape(test_input[i][:])[0], 1))
            target = test_target[i]
            output = NeuralNetwork.forward_propagate(self, x)
            prediction = np.argmax(output, axis=0)
            if i%1000 == 0:
                print(f"x: {x}\ntarget: {target}\noutput: {output}\nprediction: {prediction}")
            
            if (target == prediction):
                positive += 1
            else:
                negative += 1

        accuracy = (positive / test_size) * 100

        return accuracy

    def train(self, train_inputs, train_targets, test_inputs, test_targets , epochs):
        """Train the Neural Network."""

        epoch = 1
        train_size = np.size(train_inputs, 0)
        Train_accuracies = []
        Test_accuracies = []

        while (epoch <= epochs):
            for i in range(train_size):
                x = train_inputs[i]#[:].reshape((np.shape(train_inputs[i][:])[0], 1))
                output = NeuralNetwork.forward_propagate(self, x)
                target_vector = output #NeuralNetwork.create_target_vector(train_targets[i])
                
                NeuralNetwork.back_propagate_error(self, target_vector)
                NeuralNetwork.update_network(self)

            train_accuracy = NeuralNetwork.evaluate(self, train_inputs, train_targets)
            test_accuracy = NeuralNetwork.evaluate(self, test_inputs, test_targets)

            print(f'Epoch: [{epoch}]\nTrain accuracy: {train_accuracy}%\nTest accuracy: {test_accuracy}%')
            print('-------------------------------------------------')

            Train_accuracies.append(train_accuracy)
            Test_accuracies.append(test_accuracy)
            epoch +=1

        return Train_accuracies, Test_accuracies
