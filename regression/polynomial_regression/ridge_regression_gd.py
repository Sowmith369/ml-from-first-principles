import numpy as np

class RidgeregressionGD:
    def __init__(self, lr = 0.01, epochs=1000, lambda_=1.0):
        self.lr = lr
        self.e = epochs
        self.lambda_ = lambda_
        self.w = None
        self.b = None

    def fit(self, X ,y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.e):
            y_pred = X @ self.w + self.b

            dw = (-2 / n_samples) * (X.T @ (y-y_pred)) \
                    + 2 * self.lambda_ * self.w
            db = (-2 / n_samples) *np.sum(y - y_pred)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return X @ self.w + self.b
