import numpy as np


class ActivationFunctions:
    '''
    ActivationFunctions
    -------------------
    Provides the activation functions and respective derivatives required for 
    operating a Neural Network.

    Available activation functions:
        - sigmoid
        - tanh
        - ReLU 
        - leaky_ReLU 

    '''

    def __init__(self) -> None:
        pass

    @staticmethod
    def construct_activation_pair(activation_name: str):
        func_set = {
            'sigmoid': [ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_deriv],
            'tanh': [ActivationFunctions.tanh, ActivationFunctions.tanh_deriv],
            'ReLU': [ActivationFunctions.ReLU, ActivationFunctions.ReLU_deriv],
            'leaky_ReLU': [ActivationFunctions.leaky_ReLU, ActivationFunctions.leaky_ReLU_deriv]
        }

        if (activation_name in func_set.keys()):
            return func_set[activation_name][0], func_set[activation_name][1]

        else:
            raise(ValueError(
                f"Activation function named \"{activation_name}\" is not defined."))

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

    @staticmethod
    def ReLU(z):
        '''
        Computes the ReLU function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return np.where(z > 0, z, 0)

    @staticmethod
    def ReLU_deriv(z):
        '''
        Computes the derivative of the ReLU function for a given input z.

        @params
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_ReLU(c, z):
        '''
        Computes the leaky ReLU function for a given input z.

        @params
        - c: (float) -> constant value 
        - z: (np.ndarray) -> input object (expected to be of type np.ndarray)

        @returns
        - (np.ndarray) -> function evaluated vector or object

        '''
        return np.where(z > 0, z, c * z)

    @staticmethod
    def leaky_ReLU_deriv(c, z):
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
