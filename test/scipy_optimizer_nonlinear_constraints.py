# script to study nonlinear constraints of scipy.optimize

# Reference:
# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#constrained-minimizaiton-of-multivariate-scalar-functions-minimize
# https://stackoverflow.com/questions/19843752/structure-of-inputs-to-scipy-minimize-function

# Problem:
# min 100(x1-x0^2)^2 + (1-x0)^2
# (1) x0 + 2x1 <= 1
# (2) x0^2 + x1 <= 1
# (3) x0^2 - x1 <= 1
# (4) 2x0 + x1 = 1
# (5) 0 <= x0 <= 1
# (6) -0.5 <= x1 <= 2.0

import traceback

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, SR1, BFGS


def rosen(x):
    """The Rosenbrock function."""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def cons_f(x):
    return [x[0]**2 + x[1], x[0]**2 - x[1]]


def cons_J(x):
    return [[2*x[0], 1], [2*x[0], -1]]


def cons_H(x, v):
    return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])


if __name__ == '__main__':
    try:
        bounds = Bounds([0, -0.5], [1.0, 2.0])

        linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])

        nonlinear_constraint1 = NonlinearConstraint(cons_f, [-np.inf, -np.inf], [1, 1], jac=cons_J, hess=cons_H)
        nonlinear_constraint2 = NonlinearConstraint(cons_f, [-np.inf, -np.inf], [1, 1], jac='2-point', hess=BFGS())

        x0 = np.array([0.5, 0])

        res1 = minimize(rosen, x0, method='trust-constr', jac='3-point', hess=SR1(),
                        constraints=[linear_constraint, nonlinear_constraint1],
                        options={'verbose': 1}, bounds=bounds)
        assert res1.success
        assert np.all(np.isclose([0.4149, 0.1701], res1.x, atol=0.0001))

        res2 = minimize(rosen, x0, method='trust-constr', jac='3-point', hess=SR1(),
                        constraints=[linear_constraint, nonlinear_constraint2],
                        options={'verbose': 1}, bounds=bounds)
        assert res2.success
        assert np.all(np.isclose([0.4149, 0.1701], res2.x, atol=0.0001))

        res3 = minimize(rosen, x0, method='trust-constr', jac='3-point', hess=SR1(),
                        constraints=[nonlinear_constraint1],
                        options={'verbose': 1}, bounds=bounds)
        assert res3.success
        print(res3.x)

    except Exception as err:
        print('scipy optimizer with nonlinear constraints failed: ' + str(err))
        print(traceback.format_exc())