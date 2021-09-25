import json
import numpy as np
from NeuralNetwork import NeuralNetwork

class NeuralNetworkConfig:

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def store_network_config(NeuralNet: NeuralNetwork, model_name: str) -> None:
        '''
        Used to store neural network object variables. 

        @params
        NeuralNet: (NeuralNetwork) -> NeuralNetwork object which requires variable storage
        model_name: (str) -> name of the NeuralNetwork object.

        @returns
        None
        '''
        #get neural network variables
        network_vars = vars(NeuralNet)
        #copy neural network structure data 
        network_struct = network_vars['neuralnetwork']

        #delete non Json serializable variables
        [network_vars.pop(k) for k in ['activation', 'activation_deriv', 'neuralnetwork']]

        #save neural network structure data
        np.save(f'{model_name}_data', network_struct, allow_pickle=True)

        #dump remaining neural network variables into json file
        with open(f'{model_name}_params.json', 'w') as file:
            json.dump(network_vars, file)
        
        return 

    @staticmethod
    def load_network_config(model_name:str) -> NeuralNetwork:
        '''
        Used to load neural network object variables. 

        @params
        model_name: (str) -> name of the NeuralNetwork object.

        @returns
        NN: (NeuralNetwork) -> NeuralNetwork object created with stored variables
        '''

        network_parameters = {}

        with open(f'{model_name}_params.json') as json_file:
            network_parameters = json.load(json_file)

        neural_network_struct = np.load(f'{model_name}_data.npy', allow_pickle=True)
        
        feature_count = network_parameters['feature_count'] 
        alpha = network_parameters['alpha']
        layer_dimensions= network_parameters['layer_dimensions'] 
        activation_func = network_parameters['activation_func']
        c= network_parameters['c']

        #create NeuralNetwork object
        NN = NeuralNetwork(feature_count, alpha, layer_dimensions=layer_dimensions, activation_func=activation_func, c=c)
        NN.neuralnetwork = neural_network_struct.item()

        return NN


if __name__ == '__main__':
    pass