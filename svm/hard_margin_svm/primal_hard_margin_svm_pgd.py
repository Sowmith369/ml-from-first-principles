'''
Hard Margin SVM Primal can be sovled using many way
1) Fristly we could use Soft Margin SVM and set C high, bcz. as C increases penalty on the slack technically acting as Hard Margin SVM.
2) Directly solve the minization constraints using Quadratic Programming.
** Here I used Projected Gradient Descent (PGD) **

'''
import numpy as np

class HardMarginSVM_PGD:
    """
    Hard-Margin SVM using Projected Gradient Descent (PGD).

    Objective:
        min  0.5 * ||w||^2
    Subject to:
        y_i * (w^T x_i + b) >= 1  (Strict Hard Margin)
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert target labels to {-1, +1}
        y = np.where(y <= 0, -1, 1)

        # Initialize weight vector and bias
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):

            # -------------------------------------------------------------
            # 1. Unconstrained Gradient Descent Step
            # -------------------------------------------------------------
            # Minimize objective 0.5 * ||w||^2
            # grad_w = w, grad_b = 0
            self.w -= self.lr * self.w

            # -------------------------------------------------------------
            # 2. Sequential Exact Projection Step (Half-space projection)
            # -------------------------------------------------------------
            for i in range(n_samples):
                # Calculate margin for sample i
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)

                # Check constraint violation: y_i(w^T x_i + b) < 1
                if margin < 1.0:
                    # Squared norm of extended feature vector [x_i, 1]
                    norm_sq = np.dot(X[i], X[i]) + 1.0

                    # Exact projection distance to the boundary y_i(w^T x_i + b) = 1
                    correction = (1.0 - margin) / norm_sq

                    # Update parameters to push (w, b) back into the feasible half-space
                    self.w += correction * y[i] * X[i]
                    self.b += correction * y[i]

    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        return np.where(scores >= 0, 1, -1)
