import numpy as np

from itertools import combinations_with_replacement


class PolynomialLogisticRegression:
    def __init__(self,degree=2,learning_rate=0.01,epochs=1000):
        self.degree = degree
        self.lr = learning_rate
        self.e = epochs
        self.w = None
        self.b = None


    def _polynomial_features(self, X):
      X = np.asarray(X, dtype=float)
      n_samples, n_features = X.shape

      # Collect column vectors for all degree combinations
      poly_cols = []

      for d in range(1, self.degree + 1):
          for indices in combinations_with_replacement(range(n_features), d):
              # Compute element-wise product across chosen feature columns
              poly_cols.append(np.prod(X[:, indices], axis=1))

      return np.column_stack(poly_cols)

    def sigmoid(self, z):
        return 1 / 1 + np.exp(-z)

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        n_samples, n_features = X_poly.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.e):
            z = X_poly @ self.w + self.b
            y_pred = self.sigmoid(z)

            dw = (1 / n_samples) * (X_poly.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_probability(self, X):
        X_poly = self._polynomial_features(X)
        z = X_poly @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X):
        probability = self.predict_probability(X)
        return (probability >= 0.5).astype(int)
