
import numpy as np
from lda import LDA


# Dataset with 2 features and 3 classes
X = np.array([
    [2.0, 3.0],
    [2.5, 3.5],
    [3.0, 3.0],
    [7.0, 8.0],
    [7.5, 8.5],
    [8.0, 8.0],
    [12.0, 2.0],
    [12.5, 2.5],
    [13.0, 2.0]
])

y = np.array([
    0, 0, 0,
    1, 1, 1,
    2, 2, 2
])


# Initialize LDA
lda = LDA(n_components=2)

# Fit and transform the dataset
X_transformed = lda.fit_transform(X, y)


# Display results
print("Original Data Shape:", X.shape)
print("Transformed Data Shape:", X_transformed.shape)

print("\nClass Labels:")
print(lda.classes)

print("\nWithin-Class Scatter Matrix:")
print(lda.S_w)

print("\nBetween-Class Scatter Matrix:")
print(lda.S_b)

print("\nEigenvalues:")
print(lda.eigenvalues)

print("\nProjection Matrix:")
print(lda.scalings)

print("\nTransformed Data:")
print(X_transformed)


# Verify scatter matrix identity
X_centered = X - X.mean(axis=0, keepdims=True)
S_t_direct = X_centered.T @ X_centered

print("\nTotal Scatter Matrix:")
print(lda.S_t)

print("\nScatter Identity Check:")
print(np.allclose(lda.S_t, S_t_direct))

print("\nAll transformed values are finite:")
print(np.all(np.isfinite(X_transformed)))
