import numpy as np

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.w = None
        self.b = None

    def _polynomial_features(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack(
            [X ** i for i in range(1, self.degree + 1)]
        )

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)

        X_b = np.c_[np.ones(X_poly.shape[0]), X_poly]

        theta = np.linalg.inv(X_b.T @ X_b) @X_b.T @ y

        self.b = theta[0]
        self.w = theta[1:]

    def predict(self, X):
        X_poly = self._polynomial_features(X)

        return X_poly @ self.w + self.b
