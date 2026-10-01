import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

class HardMarginDualSVM_QP:
    def __init__(self, tol=1e-5):
        self.tol = tol
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_lables = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y = y.astype(np.float64).reshape(-1,1)

        K = np.dot(X,X.T)
        P = np.outer(y,y) * K

        q = -np.ones(n_samples)

        G = -np.eye(n_samples)
        h = np.ones(n_samples)

        A = y.T
        b = np.zeros(1)

        P_cvx = matrix(P)
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
        self.support_vector_lables = y[sv_indices].ravel()

        self.w = np.dot(self.alpha * self.support_vector_lables, self.support_vectors)

        b_values = self.support_vector_lables - np.dot(self.support_vectors,self.w)
        self.b = np.mean(b_values)

    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        return np.sign(scores)
