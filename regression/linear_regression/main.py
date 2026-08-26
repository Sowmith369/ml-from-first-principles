import numpy as np


from simple_lr_closed_form import SimpleLinearRegression
from simple_lr_gd import SimpleLinearRegressionGD
from multiple_lr_closed_form import MultipleLinearRegression
from multiple_lr_gd import MultipleLinearRegressionGD

'''
Simple Linear Regression
'''

# Dataset
X = np.array([1,2,3,4,5]) auirshgawrpghariga[]
y = np.array([3,5,7,9,11])

# Create model
model = SimpleLinearRegressionGD()

# Train
model.fit(X, y)

# Predict
predictions = model.predict(X)

print("Simple Linear Regression: ")

print(" Weight (w) :",model.w)
print(" Bias (b) :",model.b)
print(" Predictions :",predictions,"\n")

'''
Multiple Linear regression
'''

# Multiple Linear Regression
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

y = np.array([3, 5, 7, 9])

model1 = MultipleLinearRegressionGD()

model1.fit(X, y)

print("Multiple Linear Regression:")

print("Weights:", model1.w)
print("Bias:", model1.b)

print(model1.predict([[5, 6]]), "\n")
