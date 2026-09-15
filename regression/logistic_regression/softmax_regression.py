from types import new_class

import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.e = epochs
        self.w = None
        self.b = None
        self.classes = None

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.w = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y] = 1

        for _ in range(self.e):

            z = X @ self.w + self.b

            y_pred = self.softmax(z)

            dw = (1 / n_samples) * (X.T @ (y_pred - Y))
            db = (1 / n_samples) * np.sum(y_pred - Y, axis=0)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_probability(self, X):

        z = X @ self.w + self.b

        return self.softmax(z)

    def predict(self, X):

        probabilities = self.predict_probability(X)

        return np.argmax(probabilities, axis=1 )
