import numpy as np

from kernel_dual_svm_qp import KernelDualSVM_QP


# ============================================================
# XOR Dataset
# ============================================================

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    -1,
     1,
     1,
    -1
])


# ============================================================
# Function to test a kernel
# ============================================================

def test_kernel(kernel, **kwargs):

    print("\n" + "=" * 60)
    print(f"KERNEL: {kernel.upper()}")
    print("=" * 60)

    model = KernelDualSVM_QP(
        C=10.0,
        kernel=kernel,
        **kwargs
    )

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = np.mean(predictions == y)

    print("\nPredictions:")
    print(predictions)

    print("\nActual:")
    print(y)

    print("\nAccuracy:")
    print(accuracy)

    print("\nAlpha:")
    print(model.alpha)

    print("\nSupport Vectors:")
    print(model.support_vectors)

    print("\nSupport Vector Labels:")
    print(model.support_vector_labels)

    print("\nBias:")
    print(model.b)

    return model


# ============================================================
# 1. Linear Kernel
# ============================================================

linear_model = test_kernel(
    kernel='linear'
)


# ============================================================
# 2. Polynomial Kernel
# ============================================================

poly_model = test_kernel(
    kernel='poly',
    degree=2,
    coef0=1.0
)


# ============================================================
# 3. RBF Kernel
# ============================================================

rbf_model = test_kernel(
    kernel='rbf',
    gamma=1.0
)


# ============================================================
# Final Comparison
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print("\nLinear predictions:")
print(linear_model.predict(X))

print("\nPolynomial predictions:")
print(poly_model.predict(X))

print("\nRBF predictions:")
print(rbf_model.predict(X))
