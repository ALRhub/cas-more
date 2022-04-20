import numpy as np
import copy
import logging
import sys


class ExperimentLogger:
    def __init__(self, objective, stop_options, log_all=False, verbosity=0):
        self.objective = objective
        self.dist = None
        self.model = None
        self.sample_db = None
        self.best_x = None
        self.best_f = -np.inf
        self.f_evals = 0
        self.iters = 0
        self.stop_dict = StopOpt(stop_options)
        self.dists = []
        self.models = []
        self.algos = []
        self.log_all = log_all

        logger = logging.getLogger('Exp Logger')
        logger.setLevel(verbosity)
        logger.addHandler(logging.StreamHandler(sys.stdout))

        self.logger = logger

    def print(self):
        f_opt = float(self.objective(self.dist.mean.T))
        self.logger.info(
            "--------------------------------- iter {} ---------------------------------------".format(self.iters))
        self.logger.info("f_opt {}".format(f_opt))
        self.logger.debug("Dist to x_opt {}".format(np.linalg.norm(self.objective._xopt - self.dist.mean.flatten())))
        dist_to_target = np.abs((self.objective._fopt - f_opt))
        self.logger.info("Dist to f_opt {}".format(dist_to_target))

    def clear(self):
        self.sample_db = None
        self.best_x = None
        self.best_f = -np.inf
        self.stop_dict.clear()

    def stop(self, check=True):
        res = self.stop_dict(self, check)  # update the stopdict and return a Dict (self)
        return res

    def update(self, dist, model, sample_db, algo=None):
        self.dist = dist
        self.model = model
        if self.log_all:
            self.dists.append(copy.copy(dist))
            self.models.append(copy.copy(model))
            if algo is not None:
                self.algos.append(copy.copy(algo))
        self.sample_db = sample_db

        new_samples = sample_db.current_samples
        new_rewards = sample_db.current_rewards
        self.f_evals += len(new_rewards)
        self.iters += 1
        f_opt_ind = np.argmax(new_rewards)
        if new_rewards[f_opt_ind] > self.best_f:
            self.best_f = new_rewards[f_opt_ind]
            self.best_x = new_samples[f_opt_ind]


standard_options = {'f_target': np.inf,
                    'max_iters': np.inf,
                    'budget': np.inf,
                    "tol_fun": 1e-12,
                    "condition_number": 1e14,
                    "entropy": 200,
                    "diverge_opt": -1e12}


class StopOpt(dict):
    def __init__(self, options: dict):  # , dist, model):
        super(StopOpt, self).__init__()
        self.options = standard_options
        filtered_options = {k: v for k, v in options.items() if v is not None}
        self.options.update(filtered_options)
        if self.options["minimize"] and not(np.isinf(self.options["f_target"])):
            self.options["f_target"] = - self.options["f_target"]
        self._stop_list = []  # to keep multiple entries

    def __call__(self, exp_logger, check=True):
        """update and return the termination conditions dictionary

        """
        if not check:
            return self
        self._update(exp_logger)
        return self

    def _update(self, exp_logger: ExperimentLogger):
        if len(exp_logger.success_hist) < 10:  # in this case termination tests fail
            return

        self.clear()
        self._addstop('f_target',
                      exp_logger.best_f > self.options['f_target'],
                      )

        self._addstop('budget',
                      exp_logger.f_evals > self.options['budget'],)

        self._addstop('max_iters',
                      exp_logger.iters > self.options['max_iters'],)

        try:
            current_range = np.max(exp_logger.sample_db.current_rewards) - np.min(exp_logger.sample_db.current_rewards)
            hist_range = np.max(exp_logger.fit_hist) - np.min(exp_logger.fit_hist)
            self._addstop('tol_fun',
                          np.abs(current_range) < self.options['tol_fun'] and np.abs(hist_range) < self.options['tol_fun'])
        except:
            pass

        self._addstop('diverge',
                      np.min(exp_logger.sample_db.current_rewards) < self.options['diverge_opt'])

        try:
            self._addstop('opt_fail',
                          not(any(exp_logger.success_hist)))
                          # np.mean(exp_logger.success_hist) < 0.9)
        except:
            pass

        self._addstop('condition_number',
                      exp_logger.dist.condition_number > self.options['condition_number'])

        self._addstop('entropy',
                      exp_logger.dist.entropy > self.options['entropy'])

        self._addstop('min_entropy',
                      exp_logger.dist.entropy < self.options['min_entropy'])

        try:
            self._addstop('final_target_hit',
                          bool(exp_logger.objective.final_target_hit))
        except:
            pass

    def _addstop(self, key, cond, val=None):
        if cond:
            self._stop_list.append(key)  # can have the same key twice
            self[key] = val if val is not None else self.options.get(key, None)

    def clear(self):
        """empty the stopdict"""
        for k in list(self):
            self.pop(k)
        self._stop_list = []
