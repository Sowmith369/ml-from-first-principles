import numpy as np

class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        if self.n_components is None:
            self.n_components = min(n_features, n_classes - 1)

        overall_mean = X.mean(axis=0, keepdims=True)

        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))

        for cls in self.classes:
            X_c = X[y == cls]
            N_c = X_c.shape[0]

            mean_c = X_c.mean(axis=0, keepdims=True)

            S_w += X_c.T @ X_c - N_c * (mean_c.T @ mean_c)

            mean_diff = mean_c - overall_mean
            S_b += N_c * (mean_diff.T @ mean_diff)

        scale = np.trace(S_w) / n_features
        regularization = 1e-8 * scale if scale > 0 else 1e-8
        S_w_reg = S_w + regularization + np.eye(n_features)

        self.S_w = S_w
        self.S_b = S_b
        self.S_t = S_w + S_b

        A = np.linalg.solve(S_w_reg, S_b)
        eigenvalues, eigenvectors = np.linalg.eig(A)

        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real

        indices = np.argsort(eigenvalues)[::-1]

        self.mean = overall_mean.ravel()
        self.eigenvalues = eigenvalues[indices]
        self.scalings = eigenvectors[:, indices[:self.n_components]]

        return self

    def transform(self, X):
        X = np.asarray(X,dtype=float)
        return (X - self.mean) @ self.scalings

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
