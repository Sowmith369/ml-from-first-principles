import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Add column of 1s for intercept
        X_b = np.c_[np.ones(X.shape[0]), X]

        # Closed-form solution
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        # Separagte intercep and weights
        self.b = theta[0]
        self.w = theta[1:]

    def predict(self, X):
        return X @ self.w + self.b
