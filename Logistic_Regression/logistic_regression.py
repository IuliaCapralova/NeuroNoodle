import numpy as np


class LogisticRegression():
    def __init__(self, n_iter: int, alpha: float, weights: np.ndarray=np.ones((1, 1)),
                 intercept: float=0.0):

        # Initialize weights, intercept, number of iter and learning rate
        self.__weights = weights
        self.__intercept = intercept
        self.n_iter = n_iter
        self.alpha = alpha

    def train(self, X: np.ndarray, y: np.ndarray):

        # number of features and data points in the dataset
        n_features = X.shape[1]
        n_data_points = X.shape[0]

        # initialize weights
        w = self._init_weights(n_features, "zeros")

        for i in range(self.n_iter):

            self.intercept = w[0]
            self.weights = w[1:]

            # Calculate predictions, gradient and loss and BCE (problems with this haha (for now))
            y_pred = self.predict(X)
            y_pred = np.array(y_pred)
            gradient_w, gradient_b = self._gradient(n_data_points, X, y, y_pred)
            BCE = self._compute_loss(X, y, y_pred)

            # Update weights
            w[1:] = w[1:] - self.alpha * gradient_w
            w[0] = w[0] - self.alpha * gradient_b

    def _init_weights(self, dim: int, typ: str) -> np.ndarray:

        if typ == "unif":
            # Initialize weights uniformly between -1 and 1
            # The size is dim + 1 to include the intercept term
            weights = np.random.uniform(-1, 1, dim + 1)
        else:
            weights = np.zeros(dim + 1)

        return weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:

        # Check dimentions of matrices. It should be:
        # Number of features = number of weights
        if X.shape[1] != len(self.weights):
            raise ValueError("Invalid dimension of weights")

        # Calculate predictions
        lin_pred_y = np.dot(X, self.weights) + self.intercept
        logist_pred_y = self._sigmoid(lin_pred_y)

        pred_list = []
        for y in logist_pred_y:
            if y <= 0.5:
                pred_list.append(0)
            else:
                pred_list.append(1)
        return pred_list

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _gradient(self, n_data_points, X, y, predictions):
        # gradient for weights
        dw = (1/n_data_points) * np.dot(X.T, (predictions - y))

        # gradient for bias
        db = (1/n_data_points) * np.sum(predictions - y)

        return dw, db

    def _compute_loss(self, X: np.ndarray,
                      y: np.ndarray, y_pred: np.ndarray) -> float:

        n_data_points = X.shape[0]

        # binary cross entropy
        epsilon = 1e-9
        loss_terms = y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)
        loss = -np.sum(loss_terms) / n_data_points
        return loss
    
    def accuracy(self, y_pred, y_truth):
        return np.sum(y_pred==y_truth)/len(y_truth)
    
    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @property
    def intercept(self) -> float:
        return self.__intercept

    @weights.setter
    def weights(self, new_weights: np.ndarray) -> None:

        self.__weights = new_weights

    @intercept.setter
    def intercept(self, new_intercept: float) -> None:

        self.__intercept = new_intercept


if __name__ == "__main__":
    print("hello")
