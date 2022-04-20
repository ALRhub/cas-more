import numpy as np
import nlopt
from types import SimpleNamespace
import logging
import sys


class MORE:
    @classmethod
    def get_default_config(cls, dim=None):

        more_config = {"epsilon_mean": 0.5,
                       "epsilon_cov": 0.005,
                       "beta_off": 0.,
                       "lambda_0": 1,
                       "nu_0": 1,
                       "use_step_size_control": True,
                       "c_sigma": 0.9,
                       "controller_p": 1,
                       "corr_coeff": 0.25,
                       "max_ent_gain": 1,
                       "max_ent_loss": 1,
                       "normed_wmd": True,
                       "path_ratio": True,
                       }

        if dim is not None:
            samples_per_iter = int(4 + np.floor(3 * np.log(dim)))

            c_sigma = 1 / (2 + dim ** 0.7)
            epsilon_cov = 1.5 / (10 + dim ** 1.5)

            more_config.update({"samples_per_iter": samples_per_iter,
                                "c_sigma": c_sigma,
                                "epsilon_cov": epsilon_cov,
                                })

        return more_config

    def __init__(self, dim: int, config_dict: dict, verbosity: int = 30):

        self.logger = logging.getLogger('MORE')
        self.logger.setLevel(verbosity)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.dim = dim
        self.options = SimpleNamespace(**config_dict)

        self.beta = None
        self.beta_off = self.options.beta_off
        self.epsilon_mean = self.options.epsilon_mean
        self.epsilon_cov = self.options.epsilon_cov
        self.lambda_0 = self.options.lambda_0
        self.nu_0 = self.options.nu_0

        # Evolution path
        self.p_sigma = np.zeros([self.dim, 1])
        self.whitened_mean_diff = np.zeros([self.dim, 1])
        self.max_ent_loss = self.options.max_ent_loss
        self.max_ent_gain = -self.options.max_ent_gain
        self.uncorr_path = 0
        self.max_path = 0
        self.norm_len = 0
        # self.set_desired_evo_path_len(self.epsilon_mean)

        # Setting up optimizer
        mean_opt = nlopt.opt(nlopt.LD_LBFGS, 1)
        mean_opt.set_lower_bounds(1e-20)
        mean_opt.set_upper_bounds(1e20)

        mean_opt.set_ftol_abs(1e-16)
        mean_opt.set_xtol_abs(1e-16)
        mean_opt.set_maxeval(500)
        mean_opt.set_maxtime(5 * 60 * 60)

        def opt_func_mean(x, grad):
            g = self._dual_function_mean(x, grad)
            if np.isinf(g):
                mean_opt.set_lower_bounds(float(x))
            return float(g.flatten())

        mean_opt.set_min_objective(opt_func_mean)

        self.mean_opt = mean_opt

        cov_opt = nlopt.opt(nlopt.LD_LBFGS, 1)

        cov_opt.set_lower_bounds((1e-20,))
        cov_opt.set_upper_bounds((1e20,))

        cov_opt.set_ftol_abs(1e-12)
        cov_opt.set_ftol_rel(1e-12)
        cov_opt.set_xtol_abs(1e-12)
        cov_opt.set_xtol_rel(1e-12)
        cov_opt.set_maxeval(500)
        cov_opt.set_maxtime(5 * 60 * 60)

        def opt_func_cov(x, grad):
            g = self._dual_function_cov(x, grad)
            if np.isinf(g):
                cov_opt.set_lower_bounds(float(x))
            return float(g.flatten())

        cov_opt.set_min_objective(opt_func_cov)

        self.cov_opt = cov_opt

        self._grad_bound_mean = 1e-5
        self._grad_bound_cov = 1e-5

        # constant values
        self._log_2_pi_k = self.dim * (np.log(2 * np.pi))
        self._entropy_const = self.dim * (np.log(2 * np.pi) + 1)

        # cached values
        self._lambda = 1
        self._nu = 1
        self._old_dist = None
        self._current_model = None
        self._old_term = None
        self._dual_mean = np.inf
        self._dual_cov = np.inf
        self._grad_mean = 0.
        self._grad_cov = 0.
        self._kl = np.inf
        self._kl_mean = np.inf
        self._kl_cov = np.inf
        self._new_entropy = np.inf
        self._entropy_diff = 0
        self._new_mean = None
        self._new_cov = None
        self._iter = 0

    def set_desired_evo_path_len(self, epsilon_mean: float):
        """
        Sets the desired target length of the evolution path
        :param epsilon_mean: bound on the mean
        :return:
        """
        if self.options.normed_wmd:
            wmd_len = epsilon_mean * 2
        else:
            wmd_len = 1

        self.norm_len = np.sqrt((1 - self.options.c_sigma) ** 2 * self.norm_len ** 2
                                + np.sqrt(self.options.c_sigma * (2 - self.options.c_sigma)) ** 2 * wmd_len
                                + 2 * (1 - self.options.c_sigma) * self.norm_len * np.sqrt(
            self.options.c_sigma * (2 - self.options.c_sigma)) * np.sqrt(wmd_len)
                                * np.cos((1 - self.options.corr_coeff) / 2 * np.pi))

    def new_natural_params_mean(self, lambda_: float):
        """
        Compute new natural parameters for the mean
        :param lambda_: Lagrangian multiplier
        :return: Linear term and precision (not the precision of the new search distribution)
        """
        lin = (lambda_ * self._old_dist.nat_mean + self._current_model.a)
        prec = (lambda_ * self._old_dist.prec + self._current_model.A)

        return lin, prec

    def new_natural_params_cov(self, nu: float):
        """
        Compute new precision
        :param nu: Laggrangian multiplier
        :return: Precision
        """
        prec = (nu * self._old_dist.prec + self._current_model.A) / nu
        return prec

    def new_dist_sigma(self, beta, old_dist):
        new_sigma = np.exp(- beta / self.dim)
        return new_sigma

    def get_beta(self, success_mean: bool):
        """
        Compute the evolution path and the resulting change in entropy
        :param success_mean: Flag whether mean update was successful (only need to compute something if True)
        :return: Entropy change beta
        """
        if success_mean and self._current_model.current_complexity == "full":
            whitened_mean_diff = self._old_dist.sqrt_prec @ (self._old_dist.mean - self._new_mean)
            self.whitened_mean_diff = whitened_mean_diff
            if self.options.normed_wmd:
                p_sigma = (1 - self.options.c_sigma) * self.p_sigma + \
                          np.sqrt(self.options.c_sigma * (2 - self.options.c_sigma)) \
                          * whitened_mean_diff / np.sqrt(2 * self.epsilon_mean)
            else:
                p_sigma = (1 - self.options.c_sigma) * self.p_sigma \
                          + np.sqrt(self.options.c_sigma * (2 - self.options.c_sigma)) * whitened_mean_diff

            len_p_sigma = np.linalg.norm(p_sigma)
            if self.options.use_step_size_control:
                if self.options.path_ratio:
                    step_size_control = self.options.controller_p * (1 - len_p_sigma / self.norm_len)
                else:
                    step_size_control = self.options.controller_p * (self.norm_len - len_p_sigma)

                beta_off = self.options.beta_off
            else:
                step_size_control = 0
                beta_off = 0
        else:
            p_sigma = (1 - self.options.c_sigma) * self.p_sigma
            step_size_control = 0
            beta_off = 0

        self.p_sigma = p_sigma

        beta = np.clip(step_size_control, self.max_ent_gain, self.max_ent_loss) + beta_off
        return beta

    def step(self, old_dist, surrogate):
        """
        Given an old distribution and a model object, perform one MORE iteration
        :param old_dist: Distribution object
        :param surrogate: quadratic model object
        :return: new distribution parameters and success variables
        """

        success_mean = False
        success_cov = False

        self._old_term = old_dist.log_det + old_dist.mean.T @ old_dist.nat_mean
        self._old_dist = old_dist
        self._current_model = surrogate

        if surrogate.current_complexity == "full":
            self.set_desired_evo_path_len(self.epsilon_mean)

        # update mean
        for i in range(10):
            self.mean_opt.set_lower_bounds(1e-20)
            lambda_, success_mean = self._dual_mean_opt()

            if success_mean:
                break

            self.lambda_0 *= 2

        if success_mean:
            if lambda_ == 1e-20:
                self.lambda_0 = self.options.lambda_0
            else:
                self.lambda_0 = self._lambda
            new_mean = self._new_mean
        else:
            self.lambda_0 = self.options.lambda_0
            new_mean = old_dist.mean

        # update cov
        for i in range(10):
            self.cov_opt.set_lower_bounds(1e-20)
            nu, success_cov = self._dual_cov_opt()

            if success_cov:
                break

            self.nu_0 *= 2

        if success_cov:
            self.nu_0 = self._nu
            new_cov = self._new_cov
        else:
            self.nu_0 = self.options.nu_0
            new_cov = old_dist.cov

        self._kl = self._kl_mean + self._kl_cov

        self.logger.debug("Change KL {}, Change Entropy {}".format(self._kl, self._entropy_diff))
        self.logger.debug("Change KL mean {}, Change KL cov {}".format(self._kl_mean, self._kl_cov))

        # update step size with evo path
        self.beta = self.get_beta(success_mean)  # set entropy constraint
        new_cov *= np.exp(- self.beta / self.dim) ** 2

        self._iter += 1

        return new_mean, new_cov, success_mean, success_cov

    def _dual_mean_opt(self):
        """
        Optimize the dual mean
        :return: Optimal Lagrangian multiplier lambda
        """
        success = False
        try:
            lambda_ = self.mean_opt.optimize([self.lambda_0])
            opt_val_mean = self.mean_opt.last_optimum_value()
            result_mean = self.mean_opt.last_optimize_result()
        except (RuntimeError, nlopt.ForcedStop, nlopt.RoundoffLimited) as e:
            self.logger.debug("Error in mean optimization: {}".format(e))
            # try to recover a solution from failed optimization
            if np.abs(self._grad_mean) < self._grad_bound_mean:
                lambda_ = self._lambda
                result_mean = 1
            else:
                lambda_ = -1
                result_mean = 5
            opt_val_mean = self._dual_mean

        except (ValueError, np.linalg.LinAlgError) as e:
            self.logger.debug("Error in mean optimization: {}".format(e))
            result_mean = 5
            opt_val_mean = self._dual_mean
            lambda_ = -1
            # self.lambda_0 *= 10

        except Exception as e:
            raise e

        finally:
            if result_mean in (1, 2, 3, 4) and ~np.isinf(opt_val_mean):
                if lambda_ <= 1e-6:
                    if self._kl_mean < 1.1 * self.epsilon_mean:
                        success = True
                else:
                    if 0.9 * self.epsilon_mean < self._kl_mean < 1.1 * self.epsilon_mean:
                        success = True

        return lambda_, success

    def _dual_cov_opt(self):
        """
        Optimize the covariance dual
        :return: Optimal Lagrangian multiplier nu
        """
        success = False
        try:
            nu = self.cov_opt.optimize([self.nu_0])
            opt_val_cov = self.cov_opt.last_optimum_value()
            result_cov = self.cov_opt.last_optimize_result()
        except (RuntimeError, nlopt.ForcedStop, nlopt.RoundoffLimited) as e:
            self.logger.debug("Error in cov optimization: {}".format(e))
            if np.abs(self._grad_cov) < self._grad_bound_cov:
                nu = self._nu
                result_cov = 1
            else:
                nu = -1
                result_cov = 5
            opt_val_cov = self._dual_cov
            # self.nu_0 *= 1.2

        except (ValueError, np.linalg.LinAlgError) as e:
            self.logger.debug("Error in cov optimization: {}".format(e))
            result_cov = 5
            opt_val_cov = self._dual_cov
            nu = -1
        finally:
            if result_cov in (1, 2, 3, 4) and ~np.isinf(opt_val_cov):
                success = False
                if nu <= 1e-7:
                    if self._kl_cov < 1.1 * self.epsilon_cov:
                        success = True
                else:
                    if 0.9 * self.epsilon_cov < self._kl_cov < 1.1 * self.epsilon_cov:
                        success = True
        return nu, success

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% dual functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def _dual_function_mean(self, x, grad):
        """
        Dual function for the mean problem
        :param x: Solution candidate
        :param grad: Gradient of the problem
        :return: Objective value
        """
        if np.any(np.isnan(x)):
            grad[:] = 0
            return np.atleast_1d(np.inf)

        lambda_ = x[0]
        self._lambda = lambda_

        new_lin, new_prec = self.new_natural_params_mean(lambda_)

        try:
            chol_new_prec = np.linalg.cholesky(new_prec)
            new_mean = np.linalg.solve(new_prec, new_lin)

            g_mu = lambda_ * self.epsilon_mean \
                + 0.5 * (new_lin.T @ new_mean - lambda_ * np.sum((self._old_dist.chol_prec.T @ self._old_dist.mean) ** 2))

            # g2 = lambda_ * self.epsilon_mean \
            #     + 0.5 * (np.sum((chol_new_prec.T @ new_mean) ** 2)
            #              - lambda_ * np.sum((self._old_dist.chol_prec.T @ self._old_dist.mean) ** 2))

            g_mu = g_mu.flatten()

            maha_dist = np.sum((self._old_dist.chol_prec.T @ (self._old_dist.mean - new_mean)) ** 2)

            d_g_d_lambda = self.epsilon_mean - 0.5 * maha_dist

            grad[:] = d_g_d_lambda

        except np.linalg.LinAlgError:
            maha_dist = 0
            new_mean = self._old_dist.mean
            g_mu = np.atleast_1d(np.inf)
            grad[:] = -0.1

        self._dual_mean = g_mu
        self._grad_mean = grad
        self._new_mean = new_mean
        self._kl_mean = 0.5 * maha_dist
        return g_mu

    def _dual_function_cov(self, x, grad):
        """
        Dual function for the covariance problem
        :param x: Solution candidate
        :param grad: Gradient of the problem
        :return: Objective value
        """
        if np.any(np.isnan(x)):
            return np.atleast_1d(np.inf)

        nu = x[0]
        self._nu = nu

        new_prec = self.new_natural_params_cov(nu)

        try:
            chol_new_prec = np.linalg.cholesky(new_prec)
            inv_chol_new_prec = np.linalg.inv(chol_new_prec)
            new_cov = inv_chol_new_prec.T @ inv_chol_new_prec
            chol_new_cov = np.linalg.cholesky(new_cov)

            # compute log(det(Sigma_p))
            new_log_det = 2 * np.sum(np.log(np.diag(chol_new_cov)))

            entropy_diff = self._old_dist.log_det - new_log_det

            g_cov = nu * self.epsilon_cov - 0.5 * nu * entropy_diff

            trace_term = np.sum(np.square(self._old_dist.chol_prec.T @ chol_new_cov))

            kl_cov = 0.5 * (trace_term - self.dim + entropy_diff)

            d_g_d_nu = self.epsilon_cov - kl_cov

        except np.linalg.LinAlgError:
            entropy_diff = 0
            kl_cov = 0
            new_cov = self._old_dist.cov
            g_cov = np.atleast_1d(np.inf)
            d_g_d_nu = 0.

        grad[0] = d_g_d_nu
        self._dual_cov = g_cov
        self._grad_cov = grad
        self._new_cov = new_cov
        self._kl_cov = kl_cov
        self._entropy_diff = entropy_diff
        return g_cov


# %%%%%%%%%%%%%%%%%%%%%%%%%% fmin function interfaces %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def fmin(objective: callable,
         x_0,
         init_sigma,
         minimize=True,
         algo_config: dict = {},
         model_config: dict = {},
         buffer_config: dict = {},
         max_iters=None,
         f_target=None,
         budget=None,
         restarts=0,
         verbosity=20,
         ):
    """
    fmin function interface
    :param objective: Callable objective function
    :param x_0: Callable or
    :param init_sigma:
    :param minimize:
    :param algo_config:
    :param model_config:
    :param buffer_config:
    :param max_iters:
    :param f_target:
    :param budget:
    :param restarts:
    :param verbosity:
    :return:
    """

    from sample_db import SimpleSampleDatabase
    from inc_compl_model import IncComplQuadModelLS
    from gauss_full_cov import GaussFullCov
    from stop_opt import ExperimentLogger
    import attrdict as ad
    from collections import deque

    if callable(x_0):
        x_start = x_0()
    else:
        x_start = x_0

    dim = len(x_start)

    buffer_config_full = SimpleSampleDatabase.get_default_config(dim)
    buffer_config_full.update(buffer_config)
    buffer_config_full = ad.AttrDict(buffer_config_full)

    model_config_full = IncComplQuadModelLS.get_default_config()
    model_config_full.update(model_config)
    model_config_full = ad.AttrDict(model_config_full)

    algo_config_full = MORE.get_default_config(dim)
    algo_config_full.update(algo_config)
    algo_config_full = ad.AttrDict(algo_config_full)

    stop_config = {"budget": budget, "max_iters": max_iters, "f_target": f_target, "minimize": minimize,
                   "min_entropy": -25 * dim}
    exp_logger = ExperimentLogger(objective, stop_config, verbosity=verbosity)

    run = 0

    while True:
        exp_logger.clear()
        exp_logger.success_hist = deque(maxlen=(int(10 + np.ceil(30 * dim / algo_config_full.samples_per_iter))))
        exp_logger.fit_hist = deque(maxlen=(int(10 + np.ceil(30 * dim / algo_config_full.samples_per_iter))))

        if callable(x_0):
            x_start = x_0()

        sample_db = SimpleSampleDatabase(buffer_config_full.max_samples)
        search_dist = GaussFullCov(x_start, init_sigma**2 * np.eye(dim))
        surrogate = IncComplQuadModelLS(dim, model_config_full)
        more = MORE(dim, algo_config_full, verbosity=verbosity)

        while not exp_logger.stop():
            new_samples = search_dist.sample(algo_config_full.samples_per_iter)

            new_rewards = np.hstack([objective(s) for s in new_samples])
            if minimize:
                # negate, MORE maximizes, but we want to minimize
                new_rewards = -new_rewards

            sample_db.add_data(new_samples, new_rewards)
            exp_logger.fit_hist.append(max(new_rewards))

            if len(sample_db.data_x) < model_config_full.min_data_frac * surrogate.model_dim:
                exp_logger.update(dist=search_dist, model=surrogate, sample_db=sample_db)
                exp_logger.success_hist.append(True)
                continue

            samples, rewards = sample_db.get_data()

            success = surrogate.learn_quad_model(samples, rewards, search_dist, )
            if not success:
                exp_logger.update(dist=search_dist, model=surrogate, sample_db=sample_db)
                exp_logger.success_hist.append(True)
                continue

            new_mean, new_cov, success_mean, success_cov = more.step(search_dist, surrogate)

            search_dist.update_params(new_mean, new_cov)

            exp_logger.update(dist=search_dist, model=surrogate, sample_db=sample_db)
            exp_logger.success_hist.append(success_mean and success_cov)

            exp_logger.print()

        run += 1

        if run > restarts or 'f_target' in exp_logger.stop(check=False) or 'final_target_hit' in exp_logger.stop(check=False)\
                or 'budget' in exp_logger.stop(check=False) or 'max_iters' in exp_logger.stop(check=False):
            break

        algo_config_full['samples_per_iter'] = 2 * algo_config_full['samples_per_iter']

        if algo_config_full.samples_per_iter >= buffer_config_full.max_samples:
            algo_config_full.max_samples = algo_config_full.samples_per_iter

    return -exp_logger.best_f if minimize else exp_logger.best_f, exp_logger.best_x
