import numpy as np
import pandas as pd


class DataProcessor:

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_data(filename: str, delimeter: str, labels: list):
        '''
        Used to load the data from a .csv file into a pandas DataFrame.

        @params
        - filename: (str) -> name of the .csv file containing all the data  
        - delimeter: (str) -> character(s) by which the data entries are seperated
        - labels: (list) -> labels for the columns in the DataFrame

        @returns
        - (pd.DataFrame) -> DataFrame containing data for the Neural Network
        '''
        return pd.read_csv(filename, sep=delimeter, names=labels)

    @staticmethod
    def split_dataset(dataset: pd.DataFrame, split_ratio: float):
        '''
        Used to generate split datasets from a given datasets based of a given split ratio.

        @params
        - dataset: (pd.DataFrame) -> original dataset to be split
        - split_ratio: (float) -> ratio of data which the second split set should receive

        @returns
        - split_set1, split_set2: (tuple[pd.DataFrame, pd.DataFrame]) -> new training and validation datasets 

        Note:

        split_ratio is the size for the validation set, i.e. if the ratio is 0.2, then the 
        ratio going to the training set is 0.8 (80%) and the validation set will 0.2 (20%) 
        of the data.
        '''
        # get training set size
        n = dataset.shape[0]

        # copy and shuffle training set
        temp_set = dataset.copy()

        # split data into respective ratios
        mid_index = int(n * (1 - split_ratio))
        split_set1 = temp_set[0: mid_index][:]  # excludes mid index
        split_set2 = temp_set[int(n * (1 - split_ratio)):][:]

        return split_set1, split_set2

    @staticmethod
    def shuffle_dataset(dataset: pd.DataFrame):
        '''
        Shuffles the entries of given dataset.

        @params
        - dataset: (pd.DataFrame) -> dataset to be shuffles

        @returns
        - (pd.DataFrame) -> shuffled version of the given dataset

        '''
        copy_set = dataset
        return copy_set.sample(frac=1, random_state=42).reset_index(drop=True)

    @staticmethod
    def normalize_dataset(dataset: pd.DataFrame):
        '''
        Normalize features in a given dataset use min-max feature scaling. 
        Data is scaled to match the range [0, 1]. 

        @params
        - dataset: (pd.DataFrame) -> given dataset to be scaled

        @returns
        - norm_set: (pd.DataFrame) -> scaled version of the given dataset
        '''
        temp_set = dataset.copy()
        scale_feature = lambda feature: (feature - feature.min()) / (feature.max() - feature.min())
        norm_set = temp_set[:].apply(scale_feature, axis=1)

        return norm_set

    @staticmethod
    def generate_minibatchs(dataset: pd.DataFrame, batch_size: int, shuffle: bool = False):
        '''
        Generates a set of mini-batches with requested batch size from a given dataset.

        @params
        - dataset: (pd.DataFrame) -> dataset from which mini-batches are generated
        - batch_size: (int) -> number of samples each mini-batch should contain
        - shuffle: (bool) -> conditional for shuffling samples of mini-batches

        @returns
        - mini_batches: (dict[int, DataFrame]) -> set of generated mini-batches
        '''

        try:
            # check if requested batch size is smaller than number of samples 
            if batch_size > dataset.shape[0]:
                raise ValueError # batch size too large, raise error
            
            # split dataset into batches 
            mini_batches = [(dataset[i: i + batch_size]).reset_index()
                        for i in range(0, dataset.shape[0], batch_size)]

            if shuffle:
                mini_batches = {i: DataProcessor.shuffle_dataset(mini_batches[i]) for i in range(len(mini_batches))}    
            else:
                mini_batches = {i: mini_batches[i] for i in range(len(mini_batches))}
            
            return mini_batches
        
        except ValueError as e:
            print(f"\nERROR:\tRequested batch size too large for this operation.\n\tNumber of samples in dataset: {dataset.shape[0]}\n\tbatch_size: {batch_size}\n")
            