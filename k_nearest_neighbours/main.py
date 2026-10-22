import numpy as np

from knn_classification import KNearestNeighboursClassification
from knn_regression import KNearestNeighboursRegression


def main():

    # -----------------------------
    # Training data
    # -----------------------------
    X_train = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],

        [8, 8],
        [8, 9],
        [9, 8],
        [9, 9]
    ])

    # 0 = first cluster
    # 1 = second cluster
    y_train = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1
    ])


    # -----------------------------
    # Test data
    # -----------------------------
    X_test = np.array([
        [1.5, 1.5],
        [2, 1],
        [8.5, 8.5],
        [9, 8]
    ])


    # -----------------------------
    # Euclidean KNN
    # -----------------------------
    print("========== Euclidean KNN ==========")

    knn_euclidean = KNearestNeighboursClassification(
        k=3,
        distance_metric='euclidean'
    )

    knn_euclidean.fit(X_train, y_train)

    predictions = knn_euclidean.predict(X_test)

    print("Test points:")
    print(X_test)

    print("\nPredictions:")
    print(predictions)


    # -----------------------------
    # Manhattan KNN
    # -----------------------------
    print("\n========== Manhattan KNN ==========")

    knn_manhattan = KNearestNeighboursClassification(
        k=3,
        distance_metric='manhattan'
    )

    knn_manhattan.fit(X_train, y_train)

    predictions = knn_manhattan.predict(X_test)

    print("Predictions:")
    print(predictions)


    # -----------------------------
    # Minkowski KNN
    # -----------------------------
    print("\n========== Minkowski KNN ==========")

    knn_minkowski = KNearestNeighboursClassification(
        k=3,
        distance_metric='minkowski',
        p=3
    )

    knn_minkowski.fit(X_train, y_train)

    predictions = knn_minkowski.predict(X_test)

    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()





# Training data
X_train = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y_train = np.array([
    10,
    20,
    30,
    40,
    50
])


# Test data
X_test = np.array([
    [2.5],
    [4.5]
])


# Create KNN Regression model
model = KNearestNeighboursRegression(
    k=3,
    distance_metric='euclidean',
    weights='uniform'
)


# Train
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


print("Test samples:")
print(X_test)

print("\nPredictions:")
print(predictions)
