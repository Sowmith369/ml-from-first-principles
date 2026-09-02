import numpy as np

from polynomial_lr_closed_form import PolynomialRegression
from polynomial_lr_gd import PolynomialRegressionGD

from ridge_regression_closed_form import RidgeRegression
from ridge_regression_gd import RidgeregressionGD
from lasso_regression_gd import LassoRegressionGD

X_multiple = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

y_multiple = np.array([5, 8, 11, 14, 17])
# =========================================================
# 5. Polynomial Regression
# =========================================================

X_poly = np.array([1, 2, 3, 4, 5])
y_poly = np.array([2, 5, 10, 17, 26])

model = PolynomialRegression(degree=2)

model.fit(X_poly, y_poly)

print("\n--- Polynomial Regression ---")
print("Weights:", model.w)
print("Bias:", model.b)
print("Prediction:", model.predict([6]))


# =========================================================
# 6. Polynomial Regression - GD
# =========================================================
# Smaller learning rate because polynomial features
# can make gradients very large.

model = PolynomialRegressionGD(
    degree=2,
    learning_rate=0.0001,
    epochs=100000
)

model.fit(X_poly, y_poly)

print("\n--- Polynomial Regression GD ---")
print("Weights:", model.w)
print("Bias:", model.b)
print("Prediction:", model.predict([6]))


# =========================================================
# 7. Ridge Regression
# =========================================================

model = RidgeRegression(lambda_=0.1)

model.fit(X_multiple, y_multiple)

print("\n--- Ridge Regression ---")
print("Weights:", model.w)
print("Bias:", model.b)
print("Prediction:", model.predict([[6, 7]]))


# =========================================================
# 8. Ridge Regression - GD
# =========================================================

model = RidgeregressionGD(
    lr=0.001,
    epochs=10000,
    lambda_=0.1
)

model.fit(X_multiple, y_multiple)

print("\n--- Ridge Regression GD ---")
print("Weights:", model.w)
print("Bias:", model.b)
print("Prediction:", model.predict([[6, 7]]))


# =========================================================
# 9. Lasso Regression - GD
# =========================================================

model = LassoRegressionGD(
    lr=0.001,
    e=10000,
    lambda_=0.1
)

model.fit(X_multiple, y_multiple)

print("\n--- Lasso Regression GD ---")
print("Weights:", model.w)
print("Bias:", model.b)
print("Prediction:", model.predict([[6, 7]]))
