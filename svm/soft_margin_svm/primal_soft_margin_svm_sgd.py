import numpy as np

class PrimalSVM_SGD:
    '''
    Primal Soft-Margin SVM using stochastic gradient descent (SGD) and subgradient on Hinge Loss.
    Solves unconstrained problem: min 0.5 * ||w||^2 + C * sum(max(0, 1-yi * (w^T.xi + b)))
    '''
    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000):
        self.C = C
        self.lr = learning_rate
        self.e = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.e):
            for i in range(n_samples):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)

                if margin < 1:
                    grad_w = self.w - self.C * y[i] * X[i]
                    grad_b = -self.C * y[i]

                else:
                    grad_w = self.w
                    grad_b = 0.0

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
