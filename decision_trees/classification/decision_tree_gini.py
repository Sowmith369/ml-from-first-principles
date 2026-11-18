from networkx.algorithms.shortest_paths import weighted
import numpy as np
from sympy.geometry.polygon import T

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.prediction = prediction

    @property
    def is_leaf(self):
        return self.prediction is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    def gini_gain(self, parent, left, right):
        n = len(parent)
        if n == 0:
            return 0.0

        n_left, n_right = len(left), len(right)
        weighted_gini = (n_left / n) * self.gini(left) + (n_right / n) * self.gini(right)
        return self.gini(parent) - weighted_gini

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_feature, best_value = None, None

        for feature in range(n_features):
            X_column = X[:, feature]

            unique_values = np.unique(X_column)
            if len(unique_values) <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X_column <= threshold
                right_mask = ~left_mask

                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = self.gini_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = threshold

        return best_feature, best_value, best_gain

    def most_common_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def grow_tree(self, X, y, depth=0):
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        if ( n_labels == 1
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return Node(prediction = self.most_common_class(y))

        best_feature, best_value, best_gain = self.best_split(X, y)

        if best_feature is None or best_gain <= 0:
            return Node(prediction=self.most_common_class(y))

        left_mask = X[:, best_feature] <= best_value
        right_mask = ~left_mask

        left_child = self.grow_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self.grow_tree(X[right_mask], y[right_mask], depth+1)

        return Node(feature=best_feature, value=best_value, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self.grow_tree(X, y, depth=0)
        return self

    def traverse_tree(self, x, node):
        if node.is_leaf:
            return node.prediction

        if x[node.feature] <= node.value:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])
