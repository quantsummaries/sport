from cvxopt import matrix
import numpy as np

from sport.functions import obj_avg_max_drawdown, obj_neg_rtrn, obj_neg_sharpe_ratio, obj_qp, obj_risk, obj_risk_parity


class Objective:
    """A factory class that generates objective functions for optimizers."""

    @classmethod
    def init_avg_max_drawdown_obj(clsc, ptf, num_days):
        """Factory method to generate maximum drawdown objective function.

        Args:
            ptf (Portfolio): a Portfolio object.
            num_days (int): number of trading days.

        Returns:
            obj (Objective): an Objective object for maximum drawdown objective function.
        """

        return Objective({'Function': obj_avg_max_drawdown, 'Params': Objective.produce_params_avg_max_drawdown(ptf, num_days)})

    @classmethod
    def init_mean_varinace_obj(cls, ptf, risk_tol):
        """Factory method to generate mean-variance objective function.

        Args:
            ptf (Portfolio): a Portfolio object.
            risk_tol (double): risk tolerance.

        Returns:
            obj (Objective): an Objective object for mean-variance objective function.
        """
        return Objective({'Function': obj_qp, 'Params': Objective.produce_params_qp(ptf=ptf, risk_tol=risk_tol)})

    @classmethod
    def init_risk_obj(cls, ptf):
        """Factory method to generate risk objective function.

        Args:
            ptf (Portfolio): a Portfolio object.

        Returns:
            obj (Objective): an Objective object for risk objective function.
        """
        return Objective({'Function': obj_risk, 'Params': Objective.produce_params_risk(ptf)})

    @classmethod
    def init_risk_parity_obj(cls, ptf):
        """Factory method to generate risk-parity objective function.

        Args:
            ptf (Portfolio): a Portfolio object.

        Returns:
            obj (Objective): an Objective object for risk objective function.
        """
        return Objective({'Function': obj_risk_parity, 'Params': Objective.produce_params_risk_parity(ptf)})

    @classmethod
    def init_rtrn_obj(cls, ptf):
        """Factory method to generate return objective function.

        Args:
            ptf (Portfolio): a Portfolio object.

        Returns:
            obj (Objective): an Objective object for return objective function.
        """
        return Objective({'Function': obj_neg_rtrn, 'Params': Objective.produce_params_rtrn(ptf)})

    @classmethod
    def init_sharpe_ratio_obj(cls, ptf, benchmark):
        """Factory method to generate Sharpe ratio objective function.

        Args:
            ptf (Portfolio): a Portfolio object.
            benchmark (double): benchmark return rate.

        Returns:
            obj (Objective): an Objective object for Sharpe ratio objective function.
        """
        return Objective({'Function': obj_neg_sharpe_ratio, 'Params': Objective.produce_params_sharpe(ptf, benchmark)})

    @classmethod
    def produce_params_avg_max_drawdown(cls, ptf, num_days):
        """Generate parameters for maximum drawdown optimization.

        Args:
            ptf (Portfolio): a Portfolio object.
            num_days (int): number of trading days.

        Returns:
            params (dict): {'RETURN': return vector of the securities, 'COVAR': covariance matrix, 'T': num_days}.
        """

        rtrn = np.array([ptf.get_security(sec_id).get_attr_value('return') for sec_id in ptf.get_id_list()])
        Q = ptf.get_covar_matrix().to_numpy().astype(np.double)

        return {'RETURN': rtrn, 'COVAR': Q, 'T': num_days}

    @classmethod
    def produce_params_qp(cls, ptf, risk_tol):
        """Generate parameters for QP optimization.

        Args:
            ptf (Portfolio): a Portfolio object.
            risk_tol (double): risk tolerance multipled to the quadratic form.

        Returns:
            params (dict): {'Q': covariance matrix (cvxopt.matrix), 'p': negative of returns (cvxopt.matrix)}.
        """
        Q = risk_tol * matrix(ptf.get_covar_matrix().to_numpy().astype(np.double))
        p = matrix([-ptf.get_security(sec_id).get_attr_value('return') for sec_id in ptf.get_id_list()])

        return {'Q': Q, 'p': p}

    @classmethod
    def produce_params_risk(cls, ptf):
        """Generate parameter for risk optimization.

        Args:
            ptf (Portfolio): a Portfolio object.

        Returns:
            params (dict): {'Q': covariance matrix (numpy.ndarray)}.
        """
        Q = ptf.get_covar_matrix().to_numpy().astype(np.double)

        return {'COVAR': Q}

    @classmethod
    def produce_params_risk_parity(cls, ptf):
        """Generate parameter for risk-parity optimization.

        Args:
            ptf (Portfolio): a Portfolio object.

        Returns:
            params (dict): {'Q': covariance matrix (numpy.ndarray)}.
        """
        Q = ptf.get_covar_matrix().to_numpy().astype(np.double)

        return {'COVAR': Q}

    @classmethod
    def produce_params_rtrn(cls, ptf):
        """Generate parameters for return objective function.

        Args:
            ptf (Portfolio): a Portfolio object.

        Returns:
            params (dict): {'RETURN': returns (numpy.array), 'NUM_VAR': number of unknown variables}.
        """
        p = np.array([ptf.get_security(sec_id).get_attr_value('return') for sec_id in ptf.get_id_list()])
        n = len(ptf.get_id_list())

        return {'RETURN': p, 'NUM_VAR': n}

    @classmethod
    def produce_params_sharpe(cls, ptf, bmk):
        """Generate parameters for Sharpe ratio objective function.

        Args:
            ptf (Portfolio): a Portfolio object.

        Returns:
            params (dict): {'Q': covar matrix (numpy.ndarray), 'p': returns (numpy.array), 'bmk': benchmark return (float)}.
        """
        Q = ptf.get_covar_matrix().to_numpy().astype(np.double)
        p = np.array([ptf.get_security(sec_id).get_attr_value('return') for sec_id in ptf.get_id_list()])

        return {'Q': Q, 'p': p, 'bmk': bmk}

    def __init__(self, args):
        """
        Args dictionary keys: {'Q', 'p'}, {'Function', 'Params'}.

        Args:
            Q (cvxopt.matrix): covariance matrix for QP optimization.
            p (cvxopt.matrix): negative of the return vector for QP optimization.
            fun (function): a generic function pointer.
            params (dict): parameters for the objective function.
        """
        if args is None:
            raise ValueError('Input args is None')
        if len(args) == 0:
            raise ValueError('Input args is empty')

        if set(args.keys()) not in [{'Q', 'p'}, {'Function', 'Params'}]:
            raise ValueError(str(set(args.keys()))) + ' is not supported'

        self._args = args

    def get_obj_cvxopt_solvers_qp(self):
        """Generate objective function for cvxopt.sovlers.qp.

        Returns:
            obj_cvxopt_solvers_qp (dict): {'Q': Q, 'p': p} or {'Function': fun, 'Params': params}.
        """
        if set(self._args.keys()) == {'Function', 'Params'}:
            value = self._args.copy()
        elif set(self._args.keys()) == {'Q', 'p'}:
            Q = self._args.get('Q')
            p = self._args.get('p')
            value = {'Function': obj_qp, 'Params': {'Q': Q, 'p': p}}
        else:
            raise ValueError(str(self._args.keys()) + ' are not supported by get_obj_cvxopt_solvers_qp()')

        return value

    def get_obj_scipy_optimize_minimize_trust_constr(self):
        """Generate objective function for scipy.optimize.minimize, method 'trust-constr'.

        Returns:
            obj_scipy_optimize_minimize_trust_constr (dict): {'Function': fun, 'Params': params}.
        """
        if set(self._args.keys()) == {'Function', 'Params'}:
            obj = self._args.copy()
        else:
            raise ValueError(str(self._args.keys()) +
                             ' are not supported by get_obj_scipy_optimize_minimize_trust_constr()')

        return obj
