import numpy as np


class GaussFullCov:
    def __init__(self, mean: np.array, cov: np.array):
        if len(mean.shape) < 2:
            mean = np.atleast_2d(mean).reshape([-1, 1])
        self.dim = mean.shape[0]
        self.mean = mean
        self.cov = cov
        self.sigma = 1

        self.prec = None  # precision of q
        self.chol_prec = None
        self.nat_mean = None  # canonical mean parameter of q
        self.log_det = None
        self.chol_cov = None
        self.sqrt_prec = None
        self.condition_number = None

        self.update_params(mean, cov)

        # a count for sample reuse with database
        self.count = 0

        # constant values
        self._log_2_pi_k = self.dim * (np.log(2 * np.pi))

    def update_params(self, mean=None, cov=None):
        if mean is not None:
            if len(mean.shape) < 2:
                mean = np.atleast_2d(mean).reshape([-1, 1])
            self.mean = mean
        if cov is not None:
            self.cov = cov
        self.chol_cov = np.linalg.cholesky(self.cov)
        inv_chol_cov = np.linalg.inv(self.chol_cov)  # cholesky of precision of q
        self.prec = inv_chol_cov.T @ inv_chol_cov
        self.chol_prec = np.linalg.cholesky(self.prec)
        self.nat_mean = self.prec @ self.mean
        self.log_det = 2 * np.sum(np.log(np.diag(self.chol_cov)))

        eig_vals, eig_vecs = np.linalg.eigh(self.cov)
        lambdas = np.real(eig_vals)
        vecs = np.real(eig_vecs)
        sqrt_l = np.sqrt(lambdas)
        # self.Q = vecs @ np.diag(lambdas ** -1) @ vecs.T
        self.eig_vals = eig_vals
        self.eig_vecs = eig_vecs
        self.sqrt_prec = vecs @ np.diag(sqrt_l ** -1) @ vecs.T
        self.condition_number = np.abs(np.max(lambdas) / np.min(lambdas))

    # only call these methods after calling update_params

    def sample(self, n_samples, row_vec=True):
        z = np.random.normal(size=(n_samples, self.dim)).T
        x = self.mean + self.chol_cov @ z
        if row_vec:
            x = x.T
        return x

    def log_pdf(self, x):
        assert x.shape[1] == self.dim
        log_pdf = - 0.5 * (self.log_det
                           + np.sum(((x - self.mean.T) @ self.chol_prec)**2, axis=1, keepdims=True)
                           + self._log_2_pi_k)

        return log_pdf

    @property
    def entropy(self):
        return 0.5 * (self.dim + self._log_2_pi_k + self.log_det)

    def kl_divergence(self, other):
        mean_div = self.mean_div(other)
        cov_div = self.cov_div(other)
        return mean_div + cov_div

    def cov_div(self, other):
        return 0.5 * (np.trace(other.prec @ self.cov) - self.dim + other.log_det - self.log_det)

    def mean_div(self, other):
        return 0.5 * np.sum((other.chol_prec.T @ (other.mean - self.mean)) ** 2)

    def shape_div(self, other):
        return 0.5 * (np.sum(np.square(other.chol_prec.T @ self.chol_cov)) - self.dim)

    def entropy_diff(self, other):
        return 0.5 * (other.log_det - self.log_det)
