import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Calculating means
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Calculate slope (W) cov(x,y) // var(x)
        self.w = (np.sum((X - x_mean) * (y-y_mean))) / np.sum((X-x_mean) ** 2)

        self.b = y_mean - self.w * x_mean

    def predict(self, X):
        return self.w * X + self.b
