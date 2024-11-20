from glis.solvers import GLIS


class GlisOptimizer:
    def __init__(self, fun, bounds, max_evals=20, options=None):
        """
        GLIS Optimizer
        :param bounds: List of tuples specifying the bounds [(low1, high1), (low2, high2), ...]
        :param max_evals: Maximum number of evaluations
        :param options: Dictionary of GLIS options (optional)
        """
        self.fun = fun
        self.bounds = bounds
        self.max_evals = max_evals
        self.options = options if options else {}

    def optimize(self, function):
        """
        Run the GLIS optimization
        :param objective: The objective function to minimize
        :return: Tuple (best_x, best_f) where best_x is the best input and best_f is the minimum value
        """
        dim = len(self.bounds)
        lb = [b[0] for b in self.bounds]
        ub = [b[1] for b in self.bounds]

        # Run GLIS
        prob = GLIS(bounds=(lb, ub), n_initial_random=10) # initialize GLIS object
        xopt, fopt = prob.solve(self.fun, self.max_evals)  # solve optimization problem
        return xopt, fopt, prob