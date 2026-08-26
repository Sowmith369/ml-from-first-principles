from re import L

import numpy as np

class MultipleLinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs = 1000):
        self.lr = learning_rate
        self.e = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.e):

            # Predictions
            y_pred = X @ self.w + self.b

            dw = (-2 / n_samples) * (X.T @ (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            self.w -= self.lr * dw
            self.b -= self.lr * db
    def predict(self, X):
        return X @ self.w + self.b
