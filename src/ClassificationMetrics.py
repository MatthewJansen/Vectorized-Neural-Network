import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from PreProcessor import DataPreProcessor
from NetworkConfigHandler import NeuralNetworkConfig


class ClassificationMetrics:
    def __init__(self) -> None:
        return

    @staticmethod
    def confusion_matrix(y, y_pred):
        '''
        Generates a confusion matrix from the given target vector y and prediction vector y_pred.

        @params
        - y: (np.ndarray) -> vector containing target values for a given dataset
        - y_pred: (np.ndarray) -> vector containing predictions made by a trined neural network

        @returns
        - conf_matrix: (pd.DataFrame) -> confusion matrix contained in a pandas dataframe with labled rows and columns
        '''
        classes = np.unique(y)
        class_count = len(classes)

        conf_matrix = pd.DataFrame(
            np.zeros((class_count, class_count), dtype=int),
            index=classes, 
            columns=classes
        )

        for true_label, pred_label in zip(y, y_pred):
            conf_matrix.loc[true_label, pred_label] += 1

        return conf_matrix

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: pd.DataFrame, model_name = '', save_fig = False):
        plt.figure(figsize=confusion_matrix.shape)
        sn.heatmap(confusion_matrix, annot=True, cmap='viridis', fmt='.8g')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        if save_fig:
            plt.savefig(f'confusion_matrix_{model_name}.png')
        plt.show()
        return


    def classification_report(y: np.ndarray, y_pred: np.ndarray):
        conf_matrix = (ClassificationMetrics.confusion_matrix(y, y_pred)).to_numpy()

        TP = np.diag(conf_matrix)
        FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
        FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
        TN = conf_matrix.sum() - (FP + FN + TP)
        
        precision = TP / (TP + FP) 
        recall = TP / (TP + FN)
        f1_score =  2 * (precision * recall) / (precision + recall)
        
        metrics = pd.DataFrame({
            'Precision': precision, 
            'Recall': recall, 
            'f1_score': f1_score
        })
        
        acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
        
        print('=================================================\n')
        print('Classification Report:\n')
        print(f'Accuracy:\t{round(acc * 100, 2)}%\n')
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 2)
        print(metrics.to_string())
        print('\n=================================================')

        report = {
            'Accuracy': acc,
            'Metrics': metrics
        }

        return report

if __name__ == '__main__':
    mnist_test = 'mnist_test.csv'
    mnist_train = 'mnist_train.csv'

    # get datasets
    delimeter = ','
    labels = ['id'] + [f'pixel_{i}' for i in range(784)]

    train_df = DataPreProcessor.load_data(mnist_train, delimeter, labels)
    test_set = DataPreProcessor.load_data(mnist_test, delimeter, labels)
    
    # decrease dataset size
    train_df = train_df[0 : int(train_df.shape[0]*1.00)][:]
    test_set = test_set[0 : int(train_df.shape[0]*1.00)][:]

    # split training data into training & validation sets
    split = 0.2
    train_set, valid_set = DataPreProcessor.split_dataset(train_df, split)
    
    # construct input output vector sets
    set_splits = lambda digit_set: (digit_set.iloc[:, 1:].to_numpy(), digit_set.iloc[:, 0].to_numpy())
    X_train, y_train = set_splits(train_set)
    X_valid, y_valid = set_splits(valid_set)
    X_test, y_test = set_splits(test_set)

    # normalize input data
    X_train = DataPreProcessor.normalize_dataset(X_train)
    X_valid = DataPreProcessor.normalize_dataset(X_valid)
    X_test = DataPreProcessor.normalize_dataset(X_test)

    #load data into neural network data structure
    model_name = 'PreTrained_Models/model7'
    NN = NeuralNetworkConfig.load_network_config(model_name)

    # compute 
    y_pred = NN.get_predictions(X_train, y_train)

    conf_matrix = ClassificationMetrics.confusion_matrix(y_train, y_pred)
    ClassificationMetrics.plot_confusion_matrix(conf_matrix)
    ClassificationMetrics.classification_report(y_train, y_pred)