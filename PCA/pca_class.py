import numpy as np
import matplotlib.pyplot as plt


class PCA_1:
    def __init__(self, k: int = None, plot: bool = False, feature_names: list = None):
        self.k_components = k
        self.plot = plot
        self.e_vectors_to_include = None
        self.explained_variance = None
        self.names = feature_names

    def standardize(self, X):
        mean = np.mean(X, axis=0) # find mean of each feature
        sd = np.std(X, axis=0, ddof=1)   # find sd of each feature
        # data centering
        X = X - mean
        X = X / sd
        return X

    def _train(self, X: np.array):
        # covarience matrix
        cov_matrix = np.dot(X.T, X) / (X.shape[0] - 1)

        # eigenvectors, eigenvalues
        eig_val, eig_vec = np.linalg.eig(cov_matrix)

        # sort indices in reversed order
        # indices_sorted = np.argsort(eig_val)[::-1]
        indices_sorted  = np.argsort(-eig_val)

        eig_val = eig_val[indices_sorted]   # sort eigenvals and eigenvecs
        eig_vec = eig_vec[:,indices_sorted]

        if self.k_components is not None:
            self.e_vectors_to_include = eig_vec[:, :self.k_components]
        else:
            self.e_vectors_to_include = eig_vec

        # find variance explained
        sum_eig_val = np.sum(eig_val)
        self.explained_variance = eig_val/ sum_eig_val

        # in case we want to plot
        if self.plot:
            self.plot_pca()
    
    def projection(self, X):
        return np.dot(X, self.e_vectors_to_include)
    
    def train_project(self, X):
        X_centered = self.standardize(X)   # data centering
        self._train(X_centered)
        projected_data = self.projection(X_centered)
        return projected_data
    
    def plot_pca(self):
        plt.figure(figsize=(15, 15))

        feature_indices = range(1, len(self.explained_variance) + 1)
        if self.names is not None:
            feature_labels = self.names
            plt.xticks(rotation=90)
        else:
            feature_labels = feature_indices
            plt.xticks(range(1, len(self.explained_variance) + 1))

        plt.bar(feature_labels, self.explained_variance, alpha=0.5, align='center', label='Individual explained variance')
        plt.step(feature_labels, np.cumsum(self.explained_variance), where='mid', label='Cumulative explained variance')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
