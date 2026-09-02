import numpy as np

class RidgeRegression:
    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        X_b = np.c_[np.ones(X.shape[0]),X]
        I = np.eye(X_b.shape[1])
        I[0,0] = 0

        theta = np.linalg.inv(X_b.T @ X_b + self.lambda_ * I) @ X_b.T @ y

        self.b = theta[0]
        self.w = theta[1:]

    def predict(self, X):
        return X @ self.w + self.b
