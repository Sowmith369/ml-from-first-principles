import numpy as np



class KNearestNeighboursClassification:
    def __init__(self, k=3, distance_metric='euclidean', p=3):
        self.k = k
        self.distance_metric = distance_metric
        self. p = p
        self._validate_params()

    def _validate_params(self):
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be a positive integer >= 1, got {self.k}.")

        valid_metrics = {'euclidean', 'manhattan', 'minkowski'}
        if self.distance_metric not in valid_metrics:
            raise ValueError(f"Invalid distance metric '{self.distance_metric}'. Valid options are {valid_metrics}.")

        if self.distance_metric == 'minkowski':
            if not isinstance(self.p, (int, float)) or self.p < 1:
                raise ValueError(f"Parameter p for Minkowski distance must be a number >= 1, got {self.p}.")

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        return self

    def compute_dist_metric(self, x):
        diff = np.abs(self.X_train - x)

        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum(diff ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(diff, axis=1)
        elif self.distance_metric == 'minkowski':
            # Formula: (sum(|x1_i - x2_i|^p))^(1/p)
            return np.sum(diff ** self.p, axis=1) ** (1.0 / self.p)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")

    def predict_single(self, x):
        distances = self.compute_dist_metric(x)

        k_idx = np.argsort(distances)[:self.k]

        k_nearest_labels = self.y_train[k_idx]

        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self.predict_single(x) for x in X])
