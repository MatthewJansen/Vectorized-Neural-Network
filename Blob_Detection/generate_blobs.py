import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

def main():
    file_name ='blobs1'
    sample_count = 10000
    COUNT = 3
    num_centers = 7
    blobs_std = 0.8

    X, y = make_blobs(n_samples=sample_count, centers=num_centers, n_features=COUNT, random_state=42, cluster_std=blobs_std)
    data = np.hstack((X, y.reshape((y.shape[0], 1))))
    heading = [f'x{i}' for i in range(X.shape[1])] + ['y']
    data = pd.DataFrame(data, columns=heading)
    print(data)
    print('=================================================')
    print(f'Initiating blobs generation...')
    data.to_csv(f'./data/{file_name}.csv', index=False)
    print(f'Process complete! Blobs generated in ../data/{file_name}.csv')
    print('=================================================')

if __name__ == '__main__':
    main()