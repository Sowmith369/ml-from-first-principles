import numpy as np

from primal_hard_margin_svm_pgd import HardMarginSVM_PGD
from dual_hard_margin_svm_qp import HardMarginDualSVM_QP


# ============================================================
# Simple Linearly Separable Dataset
# ============================================================

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

# Hard-margin SVM uses {-1, +1}
y = np.array([
    -1, -1, -1, -1,
     1,  1,  1,  1
])


# ============================================================
# 1. Hard Margin SVM - PGD
# ============================================================

print("=" * 60)
print("HARD MARGIN SVM - PGD")
print("=" * 60)

pgd_model = HardMarginSVM_PGD(
    learning_rate=0.01,
    epochs=1000
)

pgd_model.fit(X, y)

pgd_predictions = pgd_model.predict(X)

print("\nWeight (w):")
print(pgd_model.w)

print("\nBias (b):")
print(pgd_model.b)

print("\nPredictions:")
print(pgd_predictions)

print("\nActual:")
print(y)

print("\nAccuracy:")
print(np.mean(pgd_predictions == y))


# ============================================================
# 2. Hard Margin SVM - Dual QP
# ============================================================

print("\n")
print("=" * 60)
print("HARD MARGIN SVM - DUAL QP")
print("=" * 60)

qp_model = HardMarginDualSVM_QP()

qp_model.fit(X, y)

qp_predictions = qp_model.predict(X)

print("\nWeight (w):")
print(qp_model.w)

print("\nBias (b):")
print(qp_model.b)

print("\nAlpha:")
print(qp_model.alpha)

print("\nSupport Vectors:")
print(qp_model.support_vectors)

print("\nSupport Vector Labels:")
print(qp_model.support_vector_lables)

print("\nPredictions:")
print(qp_predictions)

print("\nActual:")
print(y)

print("\nAccuracy:")
print(np.mean(qp_predictions == y))


# ============================================================
# 3. Compare Both Implementations
# ============================================================

print("\n")
print("=" * 60)
print("COMPARISON")
print("=" * 60)

print("\nPGD predictions:")
print(pgd_predictions)

print("\nDual QP predictions:")
print(qp_predictions)

print("\nPredictions identical:")
print(np.array_equal(pgd_predictions, qp_predictions))
