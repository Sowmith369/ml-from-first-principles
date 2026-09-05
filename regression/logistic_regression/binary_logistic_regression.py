import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.e = epochs
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return (1 / (1 +np.exp(-z)))

    def fit(self, X, y):
        n_samples, n_features = n.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.e):
            z = X @ self.w + self.b

            y_pred = self.sigmoid(z)

            dw = (1/n_samples) * (X.T @ (y_pred - y))
            db = (1/n_samples) * (np.sum(y_pred - y))

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_probability(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X):
        probability = self.predict_probability(X)
        return (probability >= 0.5).astype(int)
