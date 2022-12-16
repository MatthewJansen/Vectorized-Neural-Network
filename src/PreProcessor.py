import numpy as np
import pandas as pd


class PreProcessor:

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_data(filename: str, column_names: list = [], delimiter: str = ','):
        '''
        Used to load the data from a .csv file into a pandas DataFrame.

        @params
        - filename: (str) -> name of the .csv file containing all the data  
        - delimiter: (str) -> character(s) by which the data entries are separated
        - column_names: (list) -> names for the columns in the DataFrame

        @returns
        - (pd.DataFrame) -> DataFrame containing data for the Neural Network
        '''
        print(f'Reading data from {filename}')
        if column_names == []:
            return pd.read_csv(filename, sep=delimiter)
        return pd.read_csv(filename, sep=delimiter, names=column_names)

    @staticmethod
    def split_dataset(dataset: pd.DataFrame, split_ratio: float, seed=42):
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
        # copy and shuffle training set
        temp_set = dataset.copy()

        # split data into respective ratios
        smaller_set = temp_set.sample(frac=split_ratio, random_state=seed)
        larger_set = temp_set.drop(smaller_set.index)

        return larger_set, smaller_set

    @staticmethod
    def shuffle_dataset(dataset: pd.DataFrame):
        '''
        Shuffles the entries of a given dataset.

        @params
        - dataset: (pd.DataFrame) -> dataset to be shuffles

        @returns
        - (pd.DataFrame) -> shuffled version of the given dataset

        '''
        copy_set = dataset.copy()
        return copy_set.sample(frac=1, random_state=42).reset_index(drop=True)

    @staticmethod
    def normalize_dataset(dataset: np.ndarray):
        '''
        Normalize features in a given dataset using min-max feature scaling. 
        Data is scaled to match the range [0, 1]. 

        @params
        - dataset: (pd.DataFrame) -> given dataset to be scaled
        
        @returns
        - norm_set: (pd.DataFrame) -> scaled version of the given dataset
        '''

        def scale_feature(feature): return (
            feature - feature.min()) / (feature.max() - feature.min())
        norm_set = np.apply_along_axis(scale_feature, 1, dataset)

        return norm_set
