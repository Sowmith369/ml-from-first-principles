import numpy as np


class PolynomialRegressionGD:
    def __init__(self, degree, learning_rate=0.01, epochs=1000):
        self.degree = degree
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def _polynomial_features(self, X):
        X = np.asarray(X, dtype=float)

        return np.column_stack(
            [X ** i for i in range(1, self.degree + 1)]
        )

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X_poly.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):

            y_pred = X_poly @ self.w + self.b

            dw = (-2 / n_samples) * (X_poly.T @ (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        X_poly = self._polynomial_features(X)

        return X_poly @ self.w + self.b
