import numpy as np

class LogisticRegressionOVR:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.e = epochs
        self.models = {}
        self.classes = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit_binary(self, X, y):
        n_samples, n_features = X.shape

        w = np.zeros(n_features)
        b = 0

        for _ in range(self.e):

            # Forward pass
            z = X @ w + b
            y_pred = self.sigmoid(z)

            # Gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update
            w -= self.lr * dw
            b -= self.lr * db

        return w, b

    def fit(self, X, y):
        self.classes = np.unique(y)

        for cls in self.classes:

            # Current class = 1
            # All other classes = 0
            y_binary = (y == cls).astype(int)

            w, b = self.fit_binary(X, y_binary)

            self.models[cls] = (w, b)

    def predict_probability(self, X):
        probabilities = []

        for cls in self.classes:
            w, b = self.models[cls]

            z = X @ w + b
            prob = self.sigmoid(z)
            probabilities.append(prob)

        return np.array(probabilities).T

    def predict(self, X):
        probabilities = self.predict_probability(X)

        return self.classes[np.argmax(probabilities, axis=1)]
