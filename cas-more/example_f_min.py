import numpy as np
from cas_more_algo import fmin
from cma.bbobbenchmarks import nfreefunclasses


dim = 10
max_iters = 1000

more_config = {
               # "corr_coeff": 0.2,
               # "c_sigma": 0.9,
               # "epsilon_cov": 0.01,
               # "epsilon_sigma": 0.001,
               "use_step_size_control": True
               }

x_start = lambda: 0.5 * np.random.randn(dim)
init_sigma = 2

# borrowing objectives from the cma package
objective = nfreefunclasses[7](0, zerof=True, zerox=False)
objective.initwithsize(curshape=(1, dim), dim=dim)

f_val, x_val = fmin(objective, x_start, init_sigma, minimize=True, algo_config=more_config,
                    verbosity=20, f_target=1e-8, budget=1e6, restarts=0)

print(f_val, x_val)
