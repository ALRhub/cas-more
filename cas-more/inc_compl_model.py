import numpy as np
from types import SimpleNamespace
import scipy.stats as sst


class IncComplQuadModelLS:
    @classmethod
    def get_default_config(cls):
        config = {
                  "whiten_input": True,
                  "normalize_features": True,
                  "normalize_output": "mean_std_robust",
                  "robust_clip_value": 3.,
                  "unnormalize_output": False,
                  "ridge_factor": 1e-8,
                  "seed": None,
                  "increase_complexity": True,
                  "min_data_frac": 1.1,
                  "use_prior": True,
                  "limit_model_opt": False,
                  "model_limit": 100,
                  "model_limit_diff": 100,
                  }

        return config

    def __init__(self, n, config_dict):
        self.n = n
        self.options = SimpleNamespace(**config_dict)

        self.data_x_org = None
        self._data_x_mean = None
        self._data_x_inv_std = None

        self.data_y_org = None
        self._data_y_mean = None
        self._data_y_std = None

        self._phi_mean = None
        self._phi_std = None

        self.data_y_min = None
        self.data_y_max = None

        self._a_quad = np.eye(self.n)
        self._a_lin = np.zeros(shape=(self.n, 1))
        self._a_0 = np.zeros(shape=(1, 1))

        self.current_complexity = None
        self.model_dim = None
        self.prior = None

        self.phi = None
        self.targets = None
        self.weights = None

        self.square_feat_lower_tri_ind = np.tril_indices(self.n)
        self._par = None
        self._last_model_opt = None

        self.dim_tri = int(self.n * (self.n + 1) / 2)

        self.model_dim_lin = 1 + self.n
        self.model_dim_diag = self.model_dim_lin + self.n
        self.model_dim_full = 1 + self.n + self.dim_tri
        self.model_dim = self.model_dim_lin
        self.current_complexity = "lin"

        self.model_params = np.zeros(self.model_dim)

        self._phi_mean = np.zeros(shape=(1, self.model_dim - 1))
        self._phi_std = np.ones(shape=(1, self.model_dim - 1))

        self.ridge_factor = self.options.ridge_factor

    @property
    def a_0(self):
        return self._a_0

    @property
    def a(self):
        return self._a_lin

    @property
    def A(self):
        return self._a_quad

    def __call__(self, x):
        return -0.5 * float(x.T @ self.A @ x + x.T @ self.a + self.a_0)

    def get_model_params(self):
        return self._a_quad, self._a_lin

    def limit_model_opt(self):
        if self.options.limit_model_opt:
            try:
                model_opt = np.linalg.solve(self._a_quad, self._a_lin)
                if self._last_model_opt is not None:
                    valid = (np.linalg.norm(model_opt - self._last_model_opt) < self.options.model_limit_diff) and \
                            (np.all(np.abs(model_opt) < self.options.model_limit))
                else:
                    valid = True
                self._last_model_opt = model_opt
            except:
                # model_opt = np.zeros_like(self._a_lin)
                valid = True

            return valid
        else:
            return True

    def fit(self):
        reg_mat = np.eye(self.model_dim)
        reg_mat[0, 0] = 0

        phi_t_phi = self.phi.T @ self.phi

        if self.options.use_prior:
            par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * reg_mat,
                                  self.phi.T @ self.targets + self.options.ridge_factor * reg_mat @ self.prior)
        else:
            par = np.linalg.solve(phi_t_phi + self.options.ridge_factor * reg_mat,
                                  self.phi.T @ self.targets)

        self._par = par

        return True

    def preprocess_data(self, data_x, data_y, dist):
        if len(data_y.shape) < 2:
            data_y = data_y[:, None]

        self._data_y_std = np.std(data_y, ddof=1)
        self._data_y_mean = np.mean(data_y)

        if self._data_y_std == 0:
            return False

        self.data_y_org = np.copy(data_y)

        try:
            data_y = self.normalize_output(data_y)
        except ValueError:
            return False

        if self.options.whiten_input:
            data_x = self.whiten_input(data_x, dist)
        else:
            self.data_x_mean = np.zeros((1, self.n))
            self.data_x_inv_std = np.eye(self.n)

        phi = self.poly_feat(data_x)

        if self.options.normalize_features:
            self.normalize_features(phi)

        self.targets = data_y
        self.phi = phi

        return True

    def postprocess_params(self):
        if self.options.normalize_features:
            par = self.denormalize_features(self._par)
        else:
            par = self._par

        if self.current_complexity == "lin":
            a_quad = np.zeros(shape=(self.n, self.n))
        elif self.current_complexity == "diag":
            a_quad = - 2 * np.diag(par[self.n + 1:].flatten())
        elif self.current_complexity == "full":
            a_quad = np.zeros((self.n, self.n))
            a_tri = par[self.n + 1:].flatten()
            a_quad[self.square_feat_lower_tri_ind] = a_tri
            a_quad = - (a_quad + a_quad.T)
        else:
            raise ValueError("Unrecognized model type")

        a_0 = par[0]
        a_lin = par[1:self.n + 1]

        if self.options.whiten_input:
            a_quad, a_lin, a_0 = self.unwhiten_params(a_quad, a_lin, a_0)
            if self.current_complexity == 'lin':
                a_quad = np.zeros(shape=(self.n, self.n))

        if self.options.unnormalize_output:
            a_quad, a_lin, a_0 = self.unnormalize_output(a_quad, a_lin, a_0)

        return a_quad, a_lin, a_0

    def poly_feat(self, data_x):
        lin_feat = data_x

        if self.current_complexity == "lin":
            phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat])
        elif self.current_complexity == "diag":
            quad_feat = lin_feat ** 2
            phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat, quad_feat])
        elif self.current_complexity == "full":
            quad_feat = np.transpose((data_x[:, :, None] @ data_x[:, None, :]),
                                     [1, 2, 0])[self.square_feat_lower_tri_ind].T
            phi = np.hstack([np.ones([data_x.shape[0], 1]), lin_feat, quad_feat])
        else:
            raise ValueError("Unrecognized model type")

        return phi

    def normalize_features(self, phi):
        phi_mean = np.mean(phi[:, 1:], axis=0, keepdims=True)
        phi_std = np.std(phi[:, 1:], axis=0, keepdims=True, ddof=1)
        phi[:, 1:] = phi[:, 1:] - phi_mean  # or only linear part? use theta_mean?
        phi[:, 1:] = phi[:, 1:] / phi_std

        self._phi_mean = phi_mean
        self._phi_std = phi_std

        return phi

    def denormalize_features(self, par):
        par[1:] = par[1:] / self._phi_std.T
        par[0] = par[0] - self._phi_mean @ par[1:]
        return par

    def _normalize_robust(self, y):
        data_y_mean = np.mean(y)
        data_y_std = np.std(y, ddof=1)
        self._data_normalizer *= data_y_std
        # self._data_y_mean = data_y_mean
        new_y = (y - data_y_mean) / data_y_std
        new_y[new_y < -self.options.robust_clip_value] = -self.options.robust_clip_value
        new_y[new_y > self.options.robust_clip_value] = self.options.robust_clip_value
        idx = (-self.options.robust_clip_value < new_y) & (new_y < self.options.robust_clip_value)
        y_tmp = new_y[idx, None]
        kurt = sst.kurtosis(y_tmp)
        if kurt > 0.55 and not np.isclose(data_y_std, 1):
            new_y[idx, None] = self._normalize_robust(y_tmp)

        new_y[new_y == -self.options.robust_clip_value] = np.min(new_y[idx])
        new_y[new_y == self.options.robust_clip_value] = np.max(new_y[idx])
        return new_y

    def normalize_output(self, y):
        norm_type = self.options.normalize_output
        if norm_type == "mean_std":
            data_y_mean = np.mean(y)
            data_y_std = np.std(y, ddof=1)
            self._data_y_mean = data_y_mean
            new_y = (y - data_y_mean) / data_y_std

        elif norm_type == "mean_std_robust":
            new_y = self._normalize_robust(y)
        else:
            raise NotImplementedError

        return new_y

    def unnormalize_output(self, a_quad, a_lin, a_0):
        norm_type = self.options.normalize_output
        if norm_type == "mean_std":
            # std_mat = np.diag(self.data_y_std)
            new_a_quad = self._data_y_std * a_quad
            new_a_lin = self._data_y_std * a_lin
            new_a_0 = self._data_y_std * a_0 + self._data_y_mean
        elif norm_type == "mean_std_robust":
            new_a_quad = self._data_normalizer * a_quad
            new_a_lin = self._data_normalizer * a_lin
            new_a_0 = a_0  # wrong but doesn't matter
        else:
            return a_quad, a_lin, a_0

        return new_a_quad, new_a_lin, new_a_0

    def whiten_input(self, x, dist):
        data_x_mean = np.mean(x, axis=0, keepdims=True)
        self.data_x_org = x
        self._data_x_mean = data_x_mean

        if self.current_complexity == "full":
            try:
                data_x_inv_std = np.linalg.inv(np.linalg.cholesky(np.cov(x, rowvar=False))).T
            except np.linalg.LinAlgError:
                data_x_inv_std = dist.sqrt_prec
            finally:
                if np.any(np.isnan(data_x_inv_std)):
                    data_x_inv_std = dist.sqrt_prec
            self._data_x_inv_std = data_x_inv_std
            x = x - data_x_mean
            x = x @ data_x_inv_std

        return x

    def unwhiten_params(self, a_quad, a_lin, a_0):
        if self.current_complexity == "full":
            a_quad = self._data_x_inv_std @ a_quad @ self._data_x_inv_std.T
            int_a_lin = self._data_x_inv_std @ a_lin
            # a_lin = - 2 * (self._data_x_mean @ int_a_quad).T + int_a_lin
            a_lin = (self._data_x_mean @ a_quad).T + int_a_lin
            a_0 = a_0 + self._data_x_mean @ (a_quad @ self._data_x_mean.T - int_a_lin)

        return a_quad, a_lin, a_0

    def update_complexity(self, data_x):
        if len(data_x) < self.options.min_data_frac * self.model_dim_lin:
            return False
        elif len(data_x) < self.options.min_data_frac * self.model_dim_diag:
            self.current_complexity = "lin"
            self.model_dim = self.model_dim_lin
            self.prior = np.zeros((1 + self.n, 1))
            return True
        elif self.options.min_data_frac * self.model_dim_diag <= len(
                data_x) < self.options.min_data_frac * self.model_dim_full:
            self.current_complexity = "diag"
            self.model_dim = self.model_dim_diag
            self.prior = np.hstack([np.zeros(1 + self.n), - 1 / np.sqrt(2 * self.n) * np.ones(self.n)])[:, None]
            return True
        else:
            self.current_complexity = "full"
            self.model_dim = self.model_dim_full
            self.prior = np.hstack(
                [np.zeros(1 + self.n), - 1 / np.sqrt(2 * self.n) * np.eye(self.n)[self.square_feat_lower_tri_ind]])[
                         :, None]
            return True

    def learn_quad_model(self, data_x, data_y, dist=None):
        sufficient_data = self.update_complexity(data_x)
        if not sufficient_data:
            return False

        self._data_normalizer = 1
        success = self.preprocess_data(data_x, data_y, dist)
        if not success:
            return False

        success = self.fit()
        if not success:
            return False

        a_quad, a_lin, a_0 = self.postprocess_params()

        self._a_quad = a_quad
        self._a_lin = a_lin
        self._a_0 = a_0
        self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])

        if self.current_complexity == "lin":
            success = True
        elif self.current_complexity == "diag":
            success = True
            self.model_params = np.vstack([np.diag(a_quad)[:, None], a_lin, a_0])
        else:
            self.model_params = np.vstack([a_quad[self.square_feat_lower_tri_ind][:, None], a_lin, a_0])
            success = self.limit_model_opt()

        return success
