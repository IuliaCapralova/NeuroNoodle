import numpy as np


class StandardScaler():
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X - self.mean, axis=0)
        return (X - self.mean) / self.scale
