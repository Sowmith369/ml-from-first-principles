import numpy as np

class Node:
    def __init__(self, feature=None, threshold = None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold

        self.left = left
        self.right = right

        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        if len(y) == 0:
            return 0

        classes, counts = np.unique(y, return_counts=True)

        probabilites = counts / len(y)

        entropy = 0

        for p in probabilites:
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def information_gain(self, parent, left, right):
        parent_entorpy = self.entropy(parent)

        n = len(parent)

        n_left = len(left)
        n_right = len(right)

        weighted_entropy = ((n_left / n) * self.entropy(left) + (n_right / n) * self.entropy(right))

        information_gain = parent_entorpy - weighted_entropy

        return information_gain

    def best_split(self, X, y):
        n_samples, n_features = X.shape

        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            values = np.unique(X[:, feature])

            for threshold in values:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                left_y = y[left_mask]
                right_y = y[right_mask]


                if len(left_y) == 0 or len(right_y) == 0 :
                    continue

                gain = self.information_gain(
                    y,
                    left_y,
                    right_y
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def most_common_class(self, y):
        classes, counts = np.unique( y, return_counts=True )
        return classes[np.argmax(counts)]

    def grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        if self.max_depth is not None:
            if depth >= self.max_depth:
                return Node(
                    value=self.most_common_class(y)
                )
        feature, threshold, gain = self.best_split(X,y)

        if feature is None or gain <= 0:
            return  Node(value=self.most_common_class(y))

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        left_child = self.grow_tree(X_left, y_left, depth+1)

        right_child = self.grow_tree(X_right, y_right, depth+1)

        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self.grow_tree(X, y, depth=0)
        return self

    def traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x,node.left)
        else:
            return self.traverse_tree(x, node.right)

    def predict(self, X):
        predictions = []

        for x in X:
            prediction = self.traverse_tree(x, self.root)
            predictions.append(prediction)

        return np.array(predictions)
