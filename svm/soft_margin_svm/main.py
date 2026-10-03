import numpy as np
from primal_soft_margin_svm_sgd import PrimalSVM_SGD


print("=" * 60)
print("SOFT MARGIN SVM using SUB-GRADIENT AND STOCHASTIC GRADIENT DESCENT:")
print("=" * 60,"\n")

# Simple binary classification dataset
X = np.array([
    [2, 3],
    [3, 4],
    [4, 5],
    [1, 1],
    [2, 1],
    [3, 2]
])

y = np.array([
    1,
    1,
    1,
    -1,
    -1,
    -1
])


# Create model
model = PrimalSVM_SGD(
    C=1.0,
    learning_rate=0.001,
    epochs=1000
)


# Train
model.fit(X, y)


# Predict
predictions = model.predict(X)



print("Weights:", model.w)
print("Bias:", model.b)

print("\nActual labels:   ", y)
print("Predicted labels:", predictions)

# Accuracy
accuracy = np.mean(predictions == np.where(y <= 0, -1, 1))
print("\nAccuracy:", accuracy)



from dual_soft_margin_svm_qp import SoftMarginDualSVM_QP

print("=" * 60)
print("SOFT MARGIN SVM using DUAL QUADRATIC PROGRAMMING:")
print("=" * 60,"\n")

# Simple binary classification dataset
X = np.array([
    [1, 2],
    [2, 3],
    [2, 1],
    [3, 2],

    [7, 8],
    [8, 9],
    [8, 7],
    [9, 8]
], dtype=float)

# SVM labels MUST be -1 and +1
y = np.array([
    -1, -1, -1, -1,
     1,  1,  1,  1
])


# Create and train SVM
model = SoftMarginDualSVM_QP(C=1.0)

model.fit(X, y)


# Predictions
predictions = model.predict(X)



print("Predictions:", predictions)
print("Actual:     ", y)

print("\nWeight (w):", model.w)
print("Bias (b):", model.b)

print("\nSupport Vectors:")
print(model.support_vectors)

print("\nSupport Vector Labels:")
print(model.support_vector_labels)

print("\nNumber of Support Vectors:",
      len(model.support_vectors))

print("\nAccuracy:",
      np.mean(predictions == y))
