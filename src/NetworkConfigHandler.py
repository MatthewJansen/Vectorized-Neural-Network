import json
import numpy as np
from NeuralNetwork import NeuralNetwork

class NetworkConfigHandler:
    """
    NeuralNetworkConfig
    -------------------
    Provides store and load features for NeuralNetwork objects.

    """
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
        print('Preparing Neural Network data for storage...')
        
        #get neural network variables
        network_vars = vars(NeuralNet)
        #copy neural network structure data 
        network_struct = network_vars['neuralnetwork']

        # copy neural network config and learning rate
        network_data = {
            'config': network_vars['config'],
            'alpha': network_vars['alpha'],
            'cost_hist': network_vars['cost_hist'],
            'accuracies': network_vars['accuracies']
        }
        
        #delete non json serializable variables
        #[network_vars.pop(k) for k in ['activation_functions', 'neuralnetwork']]

        print(f'Saving data...')
        #save neural network structure data
        np.save(f'{model_name}_data', network_struct, allow_pickle=True)

        #dump remaining neural network variables into json file
        with open(f'{model_name}_params.json', 'w') as file:
            json.dump(network_data, file)# json.dump(network_vars, file)
        print('Neural Network data stored successfully!')
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
        print('Loading Neural Network data...')
        network_parameters = {}
        # read Neural Network data from .json file
        with open(f'{model_name}_params.json') as json_file:
            network_parameters = json.load(json_file)
        
        # load Neural Network structure from numpy file
        neural_network_struct = np.load(f'{model_name}_data.npy', allow_pickle=True)
        
        # assign Neural Network variables 
        cfg = network_parameters['config']
        # feature_count = network_parameters['feature_count'] 
        alpha = network_parameters['alpha']
        # layer_dimensions = network_parameters['layer_dimensions'] 
        # activation_func = network_parameters['activation_func']
        # c = network_parameters['c']
        cost_hist = network_parameters['cost_hist']
        accuracies = network_parameters['accuracies']

        print('Re-constructing Neural Network with loaded data...')
        #create NeuralNetwork object
        NN = NeuralNetwork(cfg, alpha)
        NN.neuralnetwork = neural_network_struct.item()
        NN.cost_hist = cost_hist
        NN.accuracies = accuracies
        print('Neural Network re-constructed and ready for usage!')

        return NN


if __name__ == '__main__':
    pass