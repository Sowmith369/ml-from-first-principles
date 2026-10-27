import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self. mean = None
        self.components = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

        covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        eigenvalues = np.maximum(eigenvalues, 0.0)

        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]

        total_variance = np.sum(eigenvalues)

        if total_variance > 0:
            self.explained_variance_ratio = self.eigenvalues / total_variance
        else:
            self.explained_variance_ratio = np.zeros(self.n_components)

        self.cumulative_explained_variance = np.cumsum(
            self.explained_variance_ratio
        )

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        X_transformed = np.asarray(X_transformed, dtype=float)
        return X_transformed @ self.components + self.mean
