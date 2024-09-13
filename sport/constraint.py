import re

from cvxopt import matrix
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, BFGS

from .constants import Constants
from .functions import constr_avg_max_drawdown, constr_risk


class Constraint:
    """A factory class that generates constrains for portfolio optimization."""

    @staticmethod
    def _parse_bounds(bounds):
        """Parse a bounds in string to lower and upper bounds in float.

        Args:
            bounds (str): a bounds in string.

        Returns:
            lb (float): lower bound.
            ub (float): upper bound.
        """
        try:
            lb = float(re.sub(r'\[', '', re.split(':', bounds)[0]))
            ub = float(re.sub(r'\]', '', re.split(':', bounds)[1]))
        except Exception as err:
            raise RuntimeError('failed to parse in Constraint._parse_bounds: ' + str(err))

        return lb, ub

    @staticmethod
    def _translate(cfg_data_raw, ptf):
        """Translate raw constraint data to security-level constrants.

        Args:
            cfg_data_raw (pandas.DataFrame): table of raw constraint data.
            ptf (Portfolio): a Portfolio object.

        Returns:
            cfg_data (pandas.DataFrame): table of translated constraint data.
        """
        # populate the processed constraints data
        cfg_data = list()

        # process grouping/classification attributes
        for idx in cfg_data_raw.index:
            attr = cfg_data_raw.loc[idx, 'ATTRIBUTE']
            if attr not in Constants.grouping_attr:
                continue

            type = cfg_data_raw.loc[idx, 'TYPE']

            class_ids = cfg_data_raw.loc[idx, 'SEC_ID']
            if class_ids == 'EACH':
                class_ids = '+'.join({ptf.get_security(sec_id=x).get_attr_value(attr_name=attr) for x in ptf.get_id_list()})

            for class_id in class_ids.split('+'):
                sec_id = '+'.join([x for x in ptf.get_id_list() if class_id == ptf.get_security(sec_id=x).get_attr_value(attr_name=attr)])
                val = cfg_data_raw.loc[idx, 'VALUE']
                cfg_data.append(pd.DataFrame({'TYPE': [type], 'SEC_ID': [sec_id],
                                              'ATTRIBUTE': ['WEIGHT'], 'VALUE': [val]}
                                             ))

        # process IDs equal to 'EACH'
        for idx in cfg_data_raw.index:
            if cfg_data_raw.loc[idx, 'ATTRIBUTE'] in Constants.grouping_attr:
                continue

            if cfg_data_raw.loc[idx, 'SEC_ID'] != 'EACH':
                continue

            data = dict()
            for col in cfg_data_raw.columns:
                value = cfg_data_raw.loc[idx, col]
                if col == 'SEC_ID':
                    data[col] = [x for x in ptf.get_id_list()]
                else:
                    data[col] = [value] * len(ptf.get_id_list())
            cfg_data.append(pd.DataFrame.from_dict(data))

        # process IDs not equal to 'EACH'
        for idx in cfg_data_raw.index:
            if cfg_data_raw.loc[idx, 'ATTRIBUTE'] in Constants.grouping_attr:
                continue

            if cfg_data_raw.loc[idx, 'SEC_ID'] == 'EACH':
                continue

            data = dict()
            for col in cfg_data_raw.columns:
                value = cfg_data_raw.loc[idx, col]
                if col == 'SEC_ID' and 'ALL' in value:
                    value = Constraint._process_all(value, ptf.get_id_list())
                data[col] = [value]
            cfg_data.append(pd.DataFrame.from_dict(data))

        return pd.concat(cfg_data).reset_index(drop=True)

    @classmethod
    def init_for_cvxopt_solvers_qp(cls, G, h, A, b):
        """Factory method to create Constraint object for cvxopt.solvers.qp.

        Args:
            G (cvxopt.matrix): G matrix in QP optimization's inequality constraint.
            h (cvxopt.matrix): h matrix in QP optimization's inequality constraint.
            A (cvxopt.matrix): A matrix in QP optimization's equality constraint.
            b (cvxopt.matrix): b matrix in QP optimization's equality constraint.
        """
        return Constraint(args={'G': G, 'h': h, 'A': A, 'b': b})

    @classmethod
    def init_for_scipy_optimize_minimize_trust_constr(cls, bounds, linear_constraint, nonlinear_constraint):
        """Factory method to create Constraint object for scipy.optimize.minimize, method 'trust-constr'.

        Args:
            bounds (scipy.optimize.Bounds): bounds used by scipy.optimize.
            linear_constraint (scipy.optimize.LinearConstraint): linear constraints used by scipy.optimize.
            nonlinear_constraint (scipy.optimize.NonlinearConstraint): nonlinear constraints used by scipy.optimize.
        """
        return Constraint(args={'Bounds': bounds,
                                'LinearConstraint': linear_constraint,
                                'NonlinearConstraint': nonlinear_constraint})

    @classmethod
    def init_from_table(cls, ptf, data, data_format):
        """Factory method to create Constraint object from constraints table.

        Args:
            ptf (Portfolio): a Portfolio object.
            data (str/pandas.DataFrame): path to the constraint file or a data frame.
            data_format (str): 'csv', 'json', 'dataframe'.
        """
        return Constraint(args={'Portfolio': ptf, 'Data': data, 'DataFormat': data_format})

    def __init__(self, args):
        """ Low level constructor with args (dict) keys: {'Portfolio', 'Data', 'DataFormat'}, {'G', 'h', 'A', 'b'}, {'Bounds', 'LinearConstraint', 'NonlinearConstraint'}.
        """
        if args is None:
            raise ValueError('Input args is None')
        if len(args) == 0:
            raise ValueError('Input args is empty')

        self._args = args

        if set(self._args.keys()) == {'Portfolio', 'Data', 'DataFormat'}:
            ptf = self._args.get('Portfolio')
            data = self._args.get('Data')
            data_format = self._args.get('DataFormat')

            self._cfg_data_raw = None      # data frame to store the raw configuration data
            self._cfg_data = None          # data frame to store the processed configuration data (ALL & EACH)
            self._sec_id_to_idx = dict()   # {security id: its index in the security list returned from Portfolio}
            self._sec_id_list = None       # security id list returned from Portfolio object
            self._ptf = None               # Portfolio object

            if ptf is None:
                raise ValueError('Input portfolio is None')
            if data is None:
                raise ValueError('Input data is None')
            if data_format is None:
                raise ValueError('Input data format is None')
            if data_format not in {'csv', 'json', 'dataframe'}:
                raise ValueError("Input data format must be 'csv' or 'json', " + data_format + " is not being supported")

            self._ptf = ptf
            self._sec_id_list = self._ptf.get_id_list()
            for idx in range(len(self._sec_id_list)):
                self._sec_id_to_idx[self._sec_id_list[idx]] = idx

            if data_format == 'csv':
                self._cfg_data_raw = pd.read_csv(data, dtype={'SEC_ID': 'str'})
            elif data_format == 'json':
                self._cfg_data_raw = pd.read_json(data)
            elif data_format == 'dataframe':
                self._cfg_data_raw = data

            if {'TYPE', 'SEC_ID', 'ATTRIBUTE', 'ATTRIBUTE_PARAMS', 'VALUE'} != set(self._cfg_data_raw.columns):
                raise ValueError('Constraints table must have columns TYPE, SEC_ID, ATTRIBUTE, ATTRIBUTE_PARAMS, VALUE.')

            self._cfg_data = Constraint._translate(self._cfg_data_raw, self._ptf)

            # validate the raw constraints data
            for idx in self._cfg_data.index:
                id = self._cfg_data.loc[idx, 'SEC_ID']
                # check all the security IDs are in Portfolio
                id_list = re.split(r'\+', id)
                for sec_id in id_list:
                    if sec_id not in self._sec_id_to_idx:
                        raise ValueError('Invalid ID in constrains file: row ' + str(idx) + ', ' + sec_id)
        elif set(self._args.keys()) in [{'G', 'h', 'A', 'b'}, {'Bounds', 'LinearConstraint', 'NonlinearConstraint'}]:
            pass
        else:
            raise ValueError(str(self._args.keys()) + ' are not supported')

    def _get_idx_from_id(self, sec_id):
        """Get the idx of a variable by its associated security ID.

        Args:
            sec_id (str): security ID.

        Returns:
            sec_idx (int): index of the variable for the security ID.
        """
        return self._sec_id_to_idx.get(sec_id)

    def _process_all(sec_id, sec_id_list):
        """Convert 'ALL' in configuration file to 'id1+id2+...'.

        Args:
            sec_id (str): a string containing 'ALL'.
            sec_id_list (list): a list of security IDs.
        Returns:
            ids (str): 'ALL' is replaced by the sum of all security IDs.
        """
        all_id = '+'.join(sec_id_list)
        return sec_id.replace('ALL', all_id)

    def get_constr_cvxopt_solvers_qp(self):
        """Get constraints for cvxopt.solvers.qp.

        Returns:
            G (cvxopt.matrix): matrix for the linear inequality (Gx <= h).
            h (cvxopt.matrix): matrix for the linear inequality (Gx <= h).
            A (cvxopt.matrix): matrix for the linear equality (Ax = b).
            b (cvxopt.matrix): matrix for the linear equality (Ax = b).
        """

        if set(self._args.keys()) == {'Portfolio', 'Data', 'DataFormat'}:
            G = pd.DataFrame(columns=self._sec_id_list)
            h = list()
            A = pd.DataFrame(columns=self._sec_id_list)
            b = list()

            idx_G = 0
            idx_A = 0
            for idx in self._cfg_data.index:
                constr_type = self._cfg_data.loc[idx, 'TYPE']
                sec_id = self._cfg_data.loc[idx, 'SEC_ID']
                attr = self._cfg_data.loc[idx, 'ATTRIBUTE']
                value = self._cfg_data.loc[idx, 'VALUE']

                # validate attr name
                if attr != 'WEIGHT' and attr not in self._ptf.get_attr_names():
                    raise ValueError(attr + ' is neither WEIGHT nor an attribute in portoflio: ' +
                                     str(self._ptf.get_attr_names()))

                if constr_type == 'LINEAR_INEQ':
                    lb, ub = Constraint._parse_bounds(value)
                    ids = re.split(r'\+', sec_id)
                    for sec_id in ids:
                        G.loc[idx_G, sec_id] = 1 if attr == 'WEIGHT' else self._ptf.get_security(sec_id).get_attr_value(attr)
                    h.append(ub)
                    idx_G += 1
                    for sec_id in ids:
                        G.loc[idx_G, sec_id] = -1 if attr == 'WEIGHT' else -self._ptf.get_security(sec_id).get_attr_value(attr)
                    h.append(-lb)
                    idx_G += 1
                elif constr_type == 'LINEAR_EQ':
                    ids = re.split(r'\+', sec_id)
                    for sec_id in ids:
                        A.loc[idx_A, sec_id] = 1 if attr == 'WEIGHT' else self._ptf.get_security(sec_id).get_attr_value(attr)
                    b.append(float(value))
                    idx_A += 1
                else:
                    raise ValueError(constr_type + ' is not supported')

            G.fillna(0, inplace=True)
            G = matrix(G.to_numpy().astype(np.double))
            h = matrix(h)
            A.fillna(0, inplace=True)
            A = matrix(A.to_numpy().astype(np.double))
            b = matrix(b)
        elif set(self._args.keys()) == {'G', 'h', 'A', 'b'}:
            G = self._args.get('G')
            h = self._args.get('h')
            A = self._args.get('A')
            b = self._args.get('b')
        else:
            raise ValueError(str(self._args.keys()) + ' are not supported')

        return {'G': G, 'h': h, 'A': A, 'b': b}

    def get_constr_scipy_optimize_minimize_trust_constr(self):
        """Get constraints for scipy.optimize.minimize, method='trust-constr'.

        Returns:
            bounds (scipy.optimize.Bounds): bounds on weights.
            linear_constraint (scipy.optimize.LinearConstraint): linear constraints.
            nonlinear_constraint (scipy.optimize.NonlinearConstraint): nonlinear constraints.
        """
        if set(self._args.keys()) == {'Portfolio', 'Data', 'DataFormat'}:
            # initial values of lower and upper bounds for the Bounds object
            lbs_bounds = [-np.inf] * len(self._sec_id_list)
            ubs_bounds = [np.inf] * len(self._sec_id_list)

            # linear constraints in the form of a matrix and its lower and upper bounds
            lbs_linear = list()
            ubs_linear = list()
            mat_linear = pd.DataFrame(columns=self._sec_id_list)
            idx_linear = 0

            # nonlinear constraints and their lower and upper bounds
            lbs_nonlinear = list()
            ubs_nonlinear = list()
            fun_nonlinear = list()
            params_nonlinear = list()

            for idx in self._cfg_data.index:
                constr_type = self._cfg_data.loc[idx, 'TYPE']
                sec_id = self._cfg_data.loc[idx, 'SEC_ID']
                attr = self._cfg_data.loc[idx, 'ATTRIBUTE']
                attr_params = self._cfg_data.loc[idx, 'ATTRIBUTE_PARAMS']
                value = self._cfg_data.loc[idx, 'VALUE']

                # validate attr name
                if attr != 'WEIGHT' and attr not in self._ptf.get_attr_names() and attr not in Constants.nonlinear_constr:
                    raise ValueError(attr + ' is neither WEIGHT nor in the attributes of the portfolio: ' +
                                     str(self._ptf.get_attr_names()) + ' nor in the attributes of nonlinear constraints: ' +
                                     str(Constants.nonlinear_constr))

                if constr_type in ('LINEAR_INEQ', 'LINEAR_EQ'):
                    if constr_type == 'LINEAR_INEQ':
                        lb, ub = Constraint._parse_bounds(value)
                    else:
                        lb, ub = float(value), float(value)
                    ids = re.split(r'\+', sec_id)

                    # update bounds for a single weight
                    if len(ids) == 1 and attr == 'WEIGHT':
                        lbs_bounds[self._get_idx_from_id(sec_id)] = max(lb, lbs_bounds[self._get_idx_from_id(sec_id)])
                        ubs_bounds[self._get_idx_from_id(sec_id)] = min(ub, ubs_bounds[self._get_idx_from_id(sec_id)])
                        continue

                    # update linear constraint matrix and its bounds
                    for sec_id in ids:
                        mat_linear.loc[idx_linear, sec_id] = 1 if attr == 'WEIGHT' else self._ptf.get_security(sec_id).get_attr_value(attr)
                    ubs_linear.append(ub)
                    lbs_linear.append(lb)
                    idx_linear += 1
                elif constr_type in ('NONLINEAR_INEQ', 'NONLINEAR_EQ'):
                    if constr_type == 'NONLINEAR_INEQ':
                        lb, ub = Constraint._parse_bounds(value)
                    else:
                        lb, ub = float(value), float(value)
                    ids = re.split(r'\+', sec_id)

                    # update nonlinear constraints and their bounds
                    if attr == 'RISK':
                        mask = [0] * len(self._args.get('Portfolio').get_id_list())
                        for sec_id in ids:
                            mask[self._get_idx_from_id(sec_id)] = 1

                        ptf = self._args.get('Portfolio')
                        params_nonlinear.append({'COVAR': ptf.get_covar_matrix().to_numpy(),
                                                 'MASK': mask.copy()})
                        fun_nonlinear.append(lambda x, p: constr_risk(x=x, params_constr=p))
                    elif attr == 'AVG_MAX_DRAWDOWN':
                        mask = [0] * len(self._args.get('Portfolio').get_id_list())
                        for sec_id in ids:
                            mask[self._get_idx_from_id(sec_id)] = 1

                        ptf = self._args.get('Portfolio')
                        params_nonlinear.append({'COVAR': ptf.get_covar_matrix().to_numpy(),
                                                 'RETURN': np.array([ptf.get_security(sec_id).get_attr_value('return') for sec_id in ptf.get_id_list()]),
                                                 'T': attr_params,
                                                 'MASK': mask.copy()})
                        fun_nonlinear.append(lambda x, p: constr_avg_max_drawdown(x=x, params_constr=p))
                    else:
                        raise Exception(attr + ' is not a supported portfolio attribute by Constraint')
                    ubs_nonlinear.append(ub)
                    lbs_nonlinear.append(lb)
                else:
                    raise ValueError(constr_type + ' is not a supported constraint type by Constraint')

            bounds = Bounds(lb=lbs_bounds, ub=ubs_bounds)

            if len(lbs_linear) > 0:
                mat_linear.fillna(0, inplace=True)
                linear_constraint = LinearConstraint(A=mat_linear.to_numpy(dtype=np.float64),
                                                     lb=lbs_linear,
                                                     ub=ubs_linear)
            else:
                linear_constraint = None

            if len(lbs_nonlinear) > 0:
                nonlinear_constraint = NonlinearConstraint(fun=lambda x: [f(x, p) for f, p in zip(fun_nonlinear, params_nonlinear)],
                                                           lb=lbs_nonlinear,
                                                           ub=ubs_nonlinear,
                                                           jac='2-point',
                                                           hess=BFGS())
            else:
                nonlinear_constraint = None
        elif set(self._args.keys()) == {'Bounds', 'LinearConstraint', 'NonlinearConstraint'}:
            bounds = self._args.get('Bounds')
            linear_constraint = self._args.get('LinearConstraint')
            nonlinear_constraint = self._args.get('NonlinearConstraint')
        else:
            raise ValueError(str(self._args.keys()) + ' are not supported')

        return {'Bounds': bounds,
                'LinearConstraint': linear_constraint,
                'NonlinearConstraint': nonlinear_constraint}

    def to_dataframe(self, raw=True):
        """Get the constraints in a data frame.

        Args:
            raw (bool): Return raw constraints if True and processed constraints if False.

        Returns:
            df (pandas.DataFrame): constraint conditions.
        """
        value = None
        if set(self._args.keys()) == {'Portfolio', 'Data', 'DataFormat'}:
            value = self._cfg_data_raw if raw else self._cfg_data

        return value
