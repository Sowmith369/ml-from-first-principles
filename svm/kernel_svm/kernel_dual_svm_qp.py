import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

class KernelDualSVM_QP:
    """
    Soft-Margin Dual SVM supporting Linear, Polynomial, and RBF Kernels using CVXOPT.
    """
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=0.5, coef0=1.0, tol=1e-5):
        self.C = C
        self.kernel_name = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol

        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None

    def _get_kernel_matrix(self, X1, X2):
        """Computes the Kernel Matrix K(X1,X2)."""
        if self.kernel_name == 'linear':
            return np.dot(X1,X2.T)

        elif self.kernel_name == 'poly':
            return (np.dot(X1,X2.T) + self.coef0) ** self.degree

        elif self.kernel_name == 'rbf':
            X1_sq = np.sum(X1**2, axis = 1, keepdims=True)
            X2_sq = np.sum(X2**2, axis = 1, keepdims=True)
            sq_dists = X1_sq + X2_sq.T - 2 * np.dot(X1, X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self.gamma * sq_dists)

        else:
            raise ValueError(f"Unsupported Kernel: {self.kernel_name}")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = y.astype(np.float64).reshape(-1,1)

        K = self._get_kernel_matrix(X, X)
        P = np.outer(y_,y_) * K

        q = -np.ones(n_samples)

        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))

        A = y_.T
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
        self.support_vector_labels = y_[sv_indices].ravel()

        free_sv = (alphas > self.tol) & (alphas < self.C - self.tol)

        if np.any(free_sv):
            free_indices = np.where(free_sv)[0]
            K_sv = K[free_indices][:, sv_indices]
            b_values = y_[free_sv].ravel() - np.dot(K_sv, self.alpha * self.support_vector_labels)
            self.b = np.mean(b_values)
        else:
            K_sv = K[sv_indices][:, sv_indices]
            b_values = self.support_vector_labels - np.dot(K_sv, self.alpha * self.support_vector_labels)
            self.b = np.mean(b_values)

    def predict(self, X):
        K_pred = self._get_kernel_matrix(X, self.support_vectors)
        scores = np.dot(K_pred, self.alpha * self.support_vector_labels) + self.b
        return np.sign(scores)
