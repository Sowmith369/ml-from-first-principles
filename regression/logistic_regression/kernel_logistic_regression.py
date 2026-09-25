import numpy as np

class KernelLogisticRegression:
    def __init__(self, C=1.0, kernel='linear', degree=3, reg_lambda=1e-4, gamma=0.5, coef0=1.0, tol=1e-5, learning_rate = 0.01, epochs=1000):
        self.C = C
        self.kernel_name = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.lr = learning_rate
        self.e = epochs

        self.X_train = None
        self.w = None
        self.b = None

    def _get_kernel_matrix(self, X1, X2):
        """Computes the Kernel Matrix K(X1,X2)."""
        if self.kernel_name == 'linear':
            return np.dot(X1,X2.T)

        elif self.kernel_name == 'poly':
            return (np.dot(X1,X2.T) + self.coef0) ** self.degree

        elif self.kernel_name == 'rbf':
            X1_sq = np.sum(X1**2, axis = 1, keepdims=True)
            X2_sq = np.sum(X2**2, axis = 1, keepdims=True)
            sq_dists = X1_sq + X2_sq.T - 2 * np.dot(X1, X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self.gamma * sq_dists)

        else:
            raise ValueError(f"Unsupported Kernel: {self.kernel_name}")

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self,X , y):
        self.X_train = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        K = self._get_kernel_matrix(self.X_train, self.X_train)

        self.w = np.zeros(n_samples)
        self.b = 0.0

        for _ in range(self.e):
            z = K @ self.w + self.b
            y_pred = self.sigmoid(z)

            dw = (1 / n_samples) * (K @ (y_pred - y)) + self.reg_lambda * (K @ self.w)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_probability(self, X):
        X = np.asarray(X, dtype=float)

        K_test = self._get_kernel_matrix(X, self.X_train)
        z = K_test @ self.w + self.b
        return self.sigmoid(z)

    def cross_entropy_loss(self, X, y):
        y_pred = self.predict_probability(X)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1-y) * np.log(1 - y_pred))

    def predict(self, X):
        return (self.predict_probability(X) >= 0.5).astype(int)
