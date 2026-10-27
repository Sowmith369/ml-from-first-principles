import numpy as np
from pca import PCA

X = np.array([
    [2, 4, 1],
    [3, 6, 2],
    [4, 8, 3],
    [5, 10, 4],
    [6, 12, 5]
])

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)

print("\nPrincipal components:")
print(pca.components)

print("\nEigenvalues:")
print(pca.eigenvalues)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio)

print("\nCumulative explained variance:")
print(pca.cumulative_explained_variance)

X_reconstructed = pca.inverse_transform(X_reduced)

print("\nReconstructed data:")
print(X_reconstructed)
