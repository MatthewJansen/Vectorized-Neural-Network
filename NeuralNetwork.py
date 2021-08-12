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
        """Computes the sigmoid function for a given input z."""
        return (1 / (1 + np.exp(-1*z)))

    @staticmethod
    def sigmoid_deriv(z):
        """Computes the derivative of the sigmoid function for a given input z."""
        sig = NeuralNetwork.sigmoid(z)
        return (sig * (1 - sig))

    @staticmethod
    def ReLU(z):
        """Computes the ReLU function for a given input z."""
        return np.where(z > 0, z, 0)

    @staticmethod
    def ReLU_deriv(z):
        """Computes the derivative of the ReLU function for a given input z."""
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_ReLU(c, z):
        """Computes the leaky ReLU function for a given input z."""
        return np.where(z > 0, z, c * z)        
    
    @staticmethod
    def leaky_ReLU_deriv(c, z):
        """Computes the derivative of the leaky ReLU function for a given input z."""
        return np.where(z > 0, 1, c)

    @staticmethod
    def tanh(z):
        """Computes the tanh function for a given input z."""
        return np.tanh(z)

    @staticmethod
    def tanh_deriv(z):
        """Computes the derivative of the tanh function for a given input z."""
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
        z[final_layer] = NeuralNetwork.compute_activation(W[final_layer], A[final_layer - 1], b[final_layer])
        next_layer = NeuralNetwork.ReLU(z[final_layer])#.sigmoid(z[final_layer])
        A[final_layer] = next_layer

        # store activations of the network
        self.neuralnetwork['activation'] = A
        self.neuralnetwork['z'] = z

        output_layer = A[max(A.keys())]

        return output_layer

