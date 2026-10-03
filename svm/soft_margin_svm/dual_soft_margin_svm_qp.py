import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = True

class SoftMarginDualSVM_QP:
    """

    """

    def __init__(self,C = 1.0, tol=1e-5):
        self.C = C
        self.tol = tol
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y = y.astype(np.float64).reshape(-1,1)

        K = np.dot(X, X.T)
        P = np.outer(y,y) * K

        q = -np.ones(n_samples)

        G_lower = -np.eye(n_samples)
        h_lower = np.zeros(n_samples)

        G_upper = np.eye(n_samples)
        h_upper = np.ones(n_samples) * self.C

        G = np.vstack((G_lower, G_upper))
        h = np.hstack((h_lower, h_upper))

        A = y.T
        b = np.zeros(1)


        P_cvx  = matrix(P)
        q_cvx = matrix(q)
        G_cvx = matrix(G)
        h_cvx = matrix(h)
        A_cvx = matrix(A)
        b_cvx = matrix(b)

        solution = solvers.qp(
            P_cvx,
            q_cvx,
            G_cvx,
            h_cvx,
            A_cvx,
            b_cvx
        )

        alphas = np.ravel(solution['x'])

        sv_indices = alphas > self.tol
        self.alpha = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices].ravel()

        self.w = np.dot(self.alpha * self.support_vector_labels, self.support_vectors)
        free_sv = (alphas > self.tol) & (alphas < self.C - self.tol)

        if np.any(free_sv):
            b_values = y[free_sv].ravel() - np.dot(X[free_sv], self.w)
            self.b = np.mean(b_values)
        else:
            b_values = self.support_vector_labels - np.dot(X[free_sv], self.w)
            self.b = np.mean(b_values)

    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        return np.sign(scores)
