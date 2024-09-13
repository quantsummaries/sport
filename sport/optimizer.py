from cvxopt import solvers
import numpy as np
from scipy.optimize import minimize

from sport.functions import obj_neg_rtrn


class Optimizer:
    """Optimization calculator."""

    def __init__(self, x0, obj, constr):
        """
        Args:
            x0 (list): initial guess.
            obj (Objective): Objective object.
            constr (Constraint): Constraint object.
        """
        if x0 is None:
            raise ValueError('Input initial guess is None')
        if obj is None:
            raise ValueError('Input Objective object is None')
        if constr is None:
            raise ValueError('Input constraint is None')

        self._x0 = x0
        self._obj = obj
        self._constr = constr

    def optimize(self, method):
        """Compute the optimized unknown variables.

        Args:
            method (str): optimization method, 'CVXOPT_SOLVERS_QP' or 'SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR'.

        Returns:
            sol (list): optimal solution.
        """
        if method is None:
            raise ValueError('Input mehtod is None')
        if method not in {'SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR', 'CVXOPT_SOLVERS_QP'}:
            raise ValueError(method + ' is not supported')

        result = None
        if method.upper() == 'CVXOPT_SOLVERS_QP':
            result = self._opt_cvxopt_solvers_qp()
        elif method.upper() == 'SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR':
            result = self._opt_scipy_optimize_minimize_trust_constr()
        else:
            raise ValueError('Optimization method ' + method + ' is not supported')

        return result

    def _opt_cvxopt_solvers_qp(self):
        """Optimize by cvxopt.solvers.qp.

        Returns:
            sol (list): optimal solution.
        """
        solvers.options['show_progress'] = False

        constr_cvxopt = self._constr.get_constr_cvxopt_solvers_qp()

        G = constr_cvxopt.get('G')
        h = constr_cvxopt.get('h')
        A = constr_cvxopt.get('A')
        b = constr_cvxopt.get('b')

        obj_cvxopt = self._obj.get_obj_cvxopt_solvers_qp()
        Q = obj_cvxopt.get('Params').get('Q')
        p = obj_cvxopt.get('Params').get('p')

        sol = solvers.qp(2*Q, p, G, h, A, b)

        return list(sol['x'])

    def _opt_scipy_optimize_minimize_trust_constr(self):
        """Optimize by scipy.optimize.minimize, method 'trust-constr'.

        Returns:
            sol (list): optimal solution.
        """
        constr_scipy = self._constr.get_constr_scipy_optimize_minimize_trust_constr()

        bounds = constr_scipy.get('Bounds')
        linear_constr = constr_scipy.get('LinearConstraint')
        nonlinear_constr = constr_scipy.get('NonlinearConstraint')

        if (linear_constr is None) and (nonlinear_constr is None):
            raise Exception('Both linear constraints and nonlinear constraints are None')
        elif nonlinear_constr is None:
            constraints = [linear_constr]
        elif linear_constr is None:
            constraints = [nonlinear_constr]
        else:
            constraints = [linear_constr, nonlinear_constr]

        obj_scipy = self._obj.get_obj_scipy_optimize_minimize_trust_constr()
        fun = obj_scipy.get('Function')
        params_obj = obj_scipy.get('Params')

        if fun.__name__ == obj_neg_rtrn.__name__:
            num_var = params_obj.get('NUM_VAR')
            if num_var is None:
                raise ValueError('Linear programming needs to know number of unknown variables')

            res = minimize(fun=fun,
                           x0=self._x0,
                           args=params_obj,
                           method='trust-constr',
                           jac='3-point',
                           hess=lambda x, v: np.zeros((num_var, num_var)),
                           bounds=bounds,
                           constraints=constraints,
                           options={'verbose': 0,
                                    'gtol': 1e-16,
                                    'xtol': 1e-16})
        else:
            res = minimize(fun=fun,
                           x0=self._x0,
                           args=params_obj,
                           method='trust-constr',
                           jac='3-point',
                           bounds=bounds,
                           constraints=constraints,
                           options={'verbose': 0,
                                    'gtol': 1e-12,
                                    'xtol': 1e-16})
        if not res.success:
            raise RuntimeError("Optimization failed; " + str(res.message))

        return res.x
