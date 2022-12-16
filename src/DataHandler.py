import numpy as np

class DataHandler:

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, shuffle:bool=False, seed:int=42) -> None:
        self.X = inputs
        self.y = targets
        self.seed = seed
        self.shuffled_indices = None

        if shuffle:
            np.random.seed(self.seed)
            self.shuffled_indices = np.arange(len(self.X))
            self.shuffled_indices = np.random.shuffle(self.shuffled_indices)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.shuffled_indices != None:
            return self.X[self.shuffled_indices[idx]], self.y[self.shuffled_indices[idx]]
        return self.X[idx], self.y[idx]