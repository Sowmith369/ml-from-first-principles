import numpy as np

class SimpleLinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n = len(X)

        # Initialize paramenters
        self.w = 0
        self.b = 0

        for _ in range(self.epochs):

            #predictions
            y_pred = self.w * X + self.b

            # Gradients
            dw = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return self.w * X + self.b
