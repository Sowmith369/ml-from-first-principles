
import numpy as np

class KNearestNeighboursRegression:
    def __init__(self, k=3, distance_metric='euclidean', p=3, weights='uniform'):
            self.k = k
            self.distance_metric = distance_metric
            self.p = p
            self.weights = weights
            self._validate_params()

    def _validate_params(self):
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be a positive integer >= 1, got {self.k}.")

        valid_metrics = {'euclidean', 'manhattan', 'minkowski'}
        if self.distance_metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{self.distance_metric}'. Valid choices: {valid_metrics}.")

        if self.distance_metric == 'minkowski':
            if not isinstance(self.p, (int, float)) or self.p < 1:
                raise ValueError(f"Parameter p for Minkowski distance must be >= 1, got {self.p}.")

        valid_weights = {'uniform', 'distance'}
        if self.weights not in valid_weights:
            raise ValueError(f"Invalid weights choice '{self.weights}'. Valid choices: {valid_weights}.")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array of continuous targets, got shape {y.shape}.")
        if X.shape[0] == 0:
            raise ValueError("Training set X cannot be empty.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}.")
        if self.k > X.shape[0]:
            raise ValueError(f"k ({self.k}) cannot exceed total training samples ({X.shape[0]}).")

        self.X_train = X
        self.y_train = y
        self.n_features_in_ = X.shape[1]
        return self

    def compute_distance(self, x):
        diff = np.abs(self.X_train - x)

        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum(diff ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(diff, axis=1)
        elif self.distance_metric == 'minkowski':
            return np.sum(diff ** self.p, axis=1) ** (1.0 / self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")

    def predict_single(self, x):
        distances = self.compute_distance(x)

        k_indices = np.argsort(distances)[:self.k]
        k_distances = distances[k_indices]
        k_targets = self.y_train[k_indices]

        if self.weights == 'uniform':
            return np.mean(k_targets)

        elif self.weights == 'distance':
            zero_dist_mask = k_distances == 0.0
            if np.any(zero_dist_mask):
                return np.mean(k_targets[zero_dist_mask])
            weights = 1.0 / k_distances
            return np.sum(weights * k_targets) / np.sum(weights)

    def predict(self, X):
        if not hasattr(self, 'X_train'):
                    raise RuntimeError("This instance is not fitted yet. Call 'fit' before calling 'predict'.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
                    raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("Test set X cannot be empty.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature mismatch: fit expected {self.n_features_in_} features, got {X.shape[1]}.")
        return np.array([self.predict_single(x) for x in X])
