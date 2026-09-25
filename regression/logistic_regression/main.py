import numpy as np

from sklearn.datasets import (
    make_classification,
    make_moons,
    load_iris
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from binary_logistic_regression import BinaryLogisticRegression
from polynomial_logistic_regression import PolynomialLogisticRegression
from kernel_logistic_regression import KernelLogisticRegression
from logistic_regression_ovr import LogisticRegressionOVR
from softmax_regression import SoftmaxRegression


# ============================================================
# 1. Binary Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("BINARY LOGISTIC REGRESSION")
print("=" * 60)

X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = BinaryLogisticRegression(
    learning_rate=0.01,
    epochs=2000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Weights:", model.w)
print("Bias:", model.b)
print("Accuracy:", accuracy_score(y_test, y_pred))


# ============================================================
# 2. Polynomial Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("POLYNOMIAL LOGISTIC REGRESSION")
print("=" * 60)

# Non-linear dataset
X, y = make_moons(
    n_samples=500,
    noise=0.15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = PolynomialLogisticRegression(
    degree=3,
    learning_rate=0.01,
    epochs=5000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Weights:", model.w)
print("Bias:", model.b)
print("Accuracy:", accuracy_score(y_test, y_pred))


# ============================================================
# 3. Kernel Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("KERNEL LOGISTIC REGRESSION")
print("=" * 60)

X, y = make_moons(
    n_samples=500,
    noise=0.15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = KernelLogisticRegression(
    kernel="rbf",
    gamma=1.0,
    reg_lambda=0.001,
    learning_rate=0.01,
    epochs=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Kernel:", model.kernel_name)
print("Accuracy:", accuracy_score(y_test, y_pred))


# ============================================================
# 4. Logistic Regression - One vs Rest
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - ONE VS REST")
print("=" * 60)

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegressionOVR(
    learning_rate=0.01,
    epochs=3000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classes:", model.classes)
print("Accuracy:", accuracy_score(y_test, y_pred))


# ============================================================
# 5. Softmax Regression
# ============================================================

print("\n" + "=" * 60)
print("SOFTMAX REGRESSION")
print("=" * 60)

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = SoftmaxRegression(
    learning_rate=0.01,
    epochs=3000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classes:", model.classes)
print("Accuracy:", accuracy_score(y_test, y_pred))


# ============================================================
# 6. Example probability prediction
# ============================================================

print("\n" + "=" * 60)
print("SOFTMAX PROBABILITY EXAMPLE")
print("=" * 60)

sample = X_test[:5]

probabilities = model.predict_probability(sample)

print("Probabilities:")
print(probabilities)

print("\nPredictions:")
print(model.predict(sample))
