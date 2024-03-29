import numpy as np


class ActivationFunctions:
    '''
    ActivationFunctions
    -------------------
    Provides the activation functions and respective derivatives required for 
    operating a Neural Network.

    Available activation functions:
        - sigmoid
        - softmax
        - tanh
        - ReLU 
        - leaky_ReLU 

    '''

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_activation_pair(activation_name: str):
        '''
        Used to construct and return a tuple containing the desired activation 
        function along with its derivative.  

        @params
        - activation_name: (str) -> name of the desired activation function for the Neural Network

        @returns
        - (tuple) -> the desired activation function along with its derivative 
        '''
        # construct dict covering all the available pair functions (f and f')
        func_set = {
            'sigmoid': [ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_deriv],
            'softmax': [ActivationFunctions.softmax, ActivationFunctions.softmax_deriv],
            'tanh': [ActivationFunctions.tanh, ActivationFunctions.tanh_deriv],
            'relu': [ActivationFunctions.relu, ActivationFunctions.relu_deriv],
            'leaky_relu': [ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_deriv]
        }

        # check if the functions requested are defined and return if true
        if (activation_name not in func_set.keys()):
            raise (ValueError(
                f"Activation function named \"{activation_name}\" is not defined. Check the spelling of activation_name or if the desired function is listed."))

        return func_set[activation_name][0], func_set[activation_name][1]

    @staticmethod
    def sigmoid(z):
        '''
        Computes the sigmoid function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return (1 / (1 + np.exp(-1*z)))

    @staticmethod
    def sigmoid_deriv(z):
        '''
        Computes the derivative of the sigmoid function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        sig = ActivationFunctions.sigmoid(z)
        return (sig * (1 - sig))

#=============================================#
#       DO NOT USE THESE FUNCTIONS!!!!        #
#=============================================#
    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z))  # stable version
        return exp_z / exp_z.sum()

    @staticmethod
    def softmax_deriv(z):
        z_ = z.reshape(-1, 1)
        return np.diagflat(z_) - np.dot(z_, z_.T)

#=============================================#

    @staticmethod
    def relu(z):
        '''
        Computes the ReLU function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return np.where(z > 0, z, 0)

    @staticmethod
    def relu_deriv(z):
        '''
        Computes the derivative of the ReLU function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_relu(z, c=0.1):
        '''
        Computes the leaky ReLU function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)
        - c: (float) -> constant value 

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return np.where(z > 0, z, c * z)

    @staticmethod
    def leaky_relu_deriv(z, c=0.1):
        '''
        Computes the derivative of the leaky ReLU function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return np.where(z > 0, 1, c)

    @staticmethod
    def tanh(z):
        '''
        Computes the tanh function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return np.tanh(z)

    @staticmethod
    def tanh_deriv(z):
        '''
        Computes the derivative of the tanh function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return 1 - (ActivationFunctions.tanh(z) ** 2)
