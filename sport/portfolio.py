import math
from numbers import Number

import numpy as np
import pandas as pd

from .functions import util_covar_to_corr_matrix, util_is_valid_covar, obj_avg_max_drawdown, obj_neg_sharpe_ratio
from .security import Security


class Portfolio:
    """A portfolio class that carries a list of Security objects and their covariance and weights."""

    def __init__(self, weights, securities, covar_matrix):
        """Construct a Portfolio object by securities and their covariance and weights.

        Args:
            weights (dict): a dictionary of {security id: weight}; weights must sum up to 1.
            securities (list): a list of Security objects.
            covar_matrix (pandas.DataFrame): covariance matrix of the securities indexed by security IDs.
        """

        self._weights = None       # dictionary of {security id: weight}
        self._securities = None    # dictionary of {security id: Security object}
        self._corr_matrix = None   # data frame of correlation matrix
        self._covar_matrix = None  # data frame of covariance matrix
        self._sec_id_list = None   # list of security IDs in the order of covar matrix index
        self._sec_id_to_idx = None # {security id: its index in the security list}

        if weights is None:
            raise ValueError('Input argument weights is None')
        if len(weights) == 0:
            raise ValueError('Input argument weights is empty')

        self._weights = weights.copy()

        if securities is None:
            raise ValueError('Input argument securities is None')
        if len(securities) == 0:
            raise ValueError('Input argument securities is empty')

        self._securities = {sec.get_id(): sec for sec in securities}
        self._sec_id_list = [sec.get_id() for sec in securities]
        self._sec_id_to_idx = dict()
        for counter, value in enumerate(self._sec_id_list):
            self._sec_id_to_idx[value] = counter

        if covar_matrix is None:
            raise ValueError('Input argument covar_matrix is None')

        self._covar_matrix = covar_matrix.copy(deep=True)

        # validate security IDs: security IDs of weights, securities, and covar matrix must match
        if set(self._weights.keys()) != set(self._sec_id_list):
            for idx in self._weights.keys():
                if idx not in self._sec_id_list:
                    raise ValueError('Security ID ' + idx + ' is in the weight dictionary but not in the security list')
            for idx in self._sec_id_list:
                if idx not in self._weights.keys():
                    raise ValueError('Security ID ' + idx + ' is not in the weight dictionary')

        # validate weights: must sum up to 1.0
        total_wt = sum([self._weights.get(sec_id) for sec_id in self._weights])
        if not math.isclose(total_wt, 1.0):
            raise ValueError('Sum of weights ' + str(total_wt) + ' != 1.0')

        if set(self._covar_matrix.columns) != set(self._sec_id_list):
            raise ValueError('Columns of the input covariance matrix do not match with security IDs.')

        valid, err_msg = util_is_valid_covar(self._covar_matrix)
        if not valid:
            raise ValueError('Input covariance matrix is not valid: ' + err_msg + '\n' + str(self._covar_matrix))

        self._corr_matrix = util_covar_to_corr_matrix(self._covar_matrix)

    def get_attr_names(self):
        """Get the names of all the attributes in this Portfolio object.

        Returns:
            attr_names (set): a set of attribute names.
        """
        attr_names = set()
        for sec in self._securities.values():
            attr_names.update(sec.get_attr_names())

        return attr_names

    def get_attr_value(self, attr_name, sec_ids=None):
        """Get portfolio level attributes, which is the weighted sum of security level attributes; non-numeric value is
        set to None.

        Args:
            attr_name (str): attribute name.

        Returns:
            value (float): the weighted sum of security level attribute values.
        """
        if attr_name.upper() not in self.get_attr_names():
            return None

        sec_ids_set = set(self.get_id_list()) if sec_ids is None else set(sec_ids)

        value = 0.0
        for sec in self._securities.values():
            sec_id = sec.get_id()
            attr_value = sec.get_attr_value(attr_name.upper())
            if sec_id in sec_ids_set and attr_value is not None:
                weight = self._weights.get(sec.get_id())
                value = value + weight * attr_value if isinstance(attr_value, Number) else None

        result = value

        return result

    def get_avg_max_drawdown(self, num_days, sec_ids=None):
        """Get portfolio level average maximum drawdown.

        Args:
            num_days (int): number of trading days.
            sec_ids (list): a selection of the assets; all if None
        Returns:
            avg_max_drawdown (double): average maximum drawdown.
        """
        if sec_ids is None:
            sec_ids = self._sec_id_list

        # check all the security IDs are in the portfolio
        for sec_id in sec_ids:
            if sec_id not in self._securities:
                raise ValueError(sec_id + ' is not in the portfolio')

        wts = self.get_weights_list(sec_ids)
        wts = [z / sum(wts) for z in wts]
        rtrns = [self._securities.get(sec_id).get_attr_value('return') for sec_id in sec_ids]
        covar = self.get_covar_matrix(sec_ids).to_numpy().astype(np.double)
        params = {'RETURN': rtrns, 'COVAR': covar, 'T': num_days}

        return obj_avg_max_drawdown(wts, params)

    def get_contr_to_risk(self, sec_id):
        """Get the contribution to risk of a single security: https://en.wikipedia.org/wiki/Risk_parity.

        Args:
            sec_id (string): security ID.

        Returns:
            ctr (double): contribution to risk.
        """
        covar = self._covar_matrix.to_numpy().astype(np.double)
        wts = np.array([self._weights.get(sec_id) for sec_id in self._sec_id_list])
        v = np.matmul(covar, wts.transpose())
        risk = self.get_risk()
        idx = self._sec_id_to_idx.get(sec_id)
        ctr = wts[idx] * v[idx] / risk

        return ctr

    def get_covar_matrix(self, sec_id_list=None):
        """Get the covariance matrix of a list of securities.

        Args:
            sec_id_list (list): list of security IDs.

        Returns:
            covar_mat (pandas.DataFrame): covariance matrix of a list of securities.
        """
        if sec_id_list is None:
            sec_id_list = self._sec_id_list

        for sec_id in sec_id_list:
            if sec_id not in self._securities:
                raise ValueError(sec_id + ' is not in the portfolio')

        return self._covar_matrix.loc[sec_id_list, sec_id_list]

    def get_id_list(self):
        """Get the security IDs of this portfolio.

        Returns:
            id_list (list): a list of security IDs.
        """
        return self._sec_id_list

    def get_risk(self, sec_ids=None):
        """Calculate the risk of a sub-portfolio.

        Args:
            sec_ids (list): a list of security IDs; default None means the whole portoflio.

        Returns:
            risk (float): risk of the sub-portfolio determined by sec_ids.
        """
        if sec_ids is None:
            sec_ids = self._sec_id_list

        # check all the security IDs are in the portfolio
        for sec_id in sec_ids:
            if sec_id not in self._securities:
                raise ValueError(sec_id + ' is not in the portfolio')

        # calculate risk as a quadratic from
        wts = np.asarray([self._weights.get(sec_id) for sec_id in sec_ids])
        wts_transpose = wts.transpose()
        covar = self._covar_matrix.loc[sec_ids, sec_ids].to_numpy(dtype=np.float64)
        risk = np.matmul(np.matmul(wts, covar), wts_transpose)
        risk = np.sqrt(risk)

        return risk

    def get_sharpe_ratio(self, bmk, sec_ids=None):
        """Calculate the Sharpe ratio of the portfolio.

        Args:
            bmk (float): benchmark rate.
            sec_ids (list): a selection of the assets; all if None
        """
        if sec_ids is None:
            sec_ids = self._sec_id_list

        # check all the security IDs are in the portfolio
        for sec_id in sec_ids:
            if sec_id not in self._securities:
                raise ValueError(sec_id + ' is not in the portfolio')

        Q = self.get_covar_matrix(sec_ids).to_numpy().astype(np.double)
        p = np.array([self.get_security(sec_id).get_attr_value('return') for sec_id in sec_ids])
        params = {'Q': Q, 'p': p, 'bmk': bmk}
        x = self.get_weights_list(sec_ids)
        x = [z/sum(x) for z in x]

        return -obj_neg_sharpe_ratio(x, params)

    def get_security(self, sec_id):
        """Get a Security object based on security ID.

        Args:
            sec_id (str): security ID.

        Returns:
            sec (Security): a Security object.
        """
        if sec_id not in self._securities:
            raise ValueError(sec_id + ' is not in the portfolio')

        return self._securities.get(sec_id)

    def get_securities(self):
        """Get securities in this Portfolio object.

        Returns:
            securities (list): securities in this Portfolio object.
        """
        return [self._securities.get(sec_id) for sec_id in self._sec_id_list]

    def get_weights_dataframe(self, sec_ids=None, num_sig_digits=3):
        """Get security weights in a data frame by security IDs.

        Args:
            sec_ids (list): IDs of a list of Security objects.
            num_sig_digits (int): number of significant digits after decimal point.

        Returns:
            weights (pandas.DataFrame): columns={sec_id, weight}; None for whole portfolio.
        """
        if sec_ids is None:
            sec_ids = self._sec_id_list

        wts = [round(self._weights.get(sec_id), num_sig_digits) for sec_id in sec_ids]

        return pd.DataFrame.from_dict({'SEC_ID': sec_ids, 'WEIGHT': wts})

    def get_weights_dict(self, sec_ids=None):
        """Get security weights in a dictionary by security IDs.

        Args:
            sec_ids (list): IDs of a list of Security objects.

        Returns:
            weights (dict): {sec_id: weight}; defualt input None for whole portfolio.
        """
        if sec_ids is None:
            sec_ids = self._sec_id_list

        for sec_id in sec_ids:
            if sec_id not in self._securities:
                raise ValueError(sec_id + ' is not in the portfolio.')

        return {sec_id: self._weights.get(sec_id) for sec_id in sec_ids}

    def get_weights_list(self, sec_ids=None):
        """Get security weights in a list by security IDs.

        Args:
            sec_ids (list): IDs of a list of Security objects.

        Returns:
            weights (list): Weights of securities; default input None for whole portfolio.
        """
        if sec_ids is None:
            sec_ids = self._sec_id_list

        for sec_id in sec_ids:
            if sec_id not in self._securities:
                raise ValueError(sec_id + ' is not in the portfolio')

        return [self._weights.get(sec_id) for sec_id in sec_ids]

    def report(self, obj_type, obj_param, addtl=None):
        """Report the portfolio for various optimization objects.

        Args:
            obj_type (string): objective type, 'MEAN_VARIANCE', 'MAX_SHARPE_RATIO', 'MAX_RETURN', 'MIN_RISK', 'RISK_PARITY'.
            obj_param (double): objective parameter (benchmark return if obj_type is 'MAX_SHARPE_RATIO', risk tolerance if obj_type is 'MEAN_VARIANCE').
            addtl (list): list of additional fields of data to be dumped into output.

        Returns:
            result (pandas.DataFrame): a data frame with corresponding security attributes.
            port_analytics (pandas.DataFrame): portfolio level analytics relevant to objective type.
        """
        df_list = list()

        for sec_id in self._sec_id_list:
            data = dict()
            sec = self._securities.get(sec_id)

            data['SEC_ID'] = sec_id
            data['SEC_NM'] = sec.get_attr_value('SEC_NM')

            if addtl is not None and len(addtl) > 0:
                for fld in set(addtl):
                    data[fld] = sec.get_attr_value(fld)

            data['WEIGHT'] = round(self._weights.get(sec_id), 4)
            data['RETURN'] = round(sec.get_attr_value('RETURN'), 4)
            data['RISK'] = round(sec.get_attr_value('RISK'), 4)

            if 'MEAN_VARIANCE' == obj_type.upper():
                pass
            elif 'MAX_SHARPE_RATIO' == obj_type.upper():
                if obj_param is None:
                    raise ValueError(obj_type + ' reporting must have benchmark rate')
                data['SHARPE_RATIO'] = round(self.get_sharpe_ratio(bmk=obj_param, sec_ids=[sec_id]), 4)
            elif 'MAX_RETURN' == obj_type.upper():
                pass
            elif 'MIN_RISK' == obj_type.upper():
                pass
            elif 'RISK_PARITY' == obj_type.upper():
                data['CTR'] = round(self.get_contr_to_risk(sec_id), 4)
            elif 'MIN_AVG_MAX_DRAWDOWN' == obj_type.upper():
                data['AVG_MAX_DRAWDOWN'] = round(self.get_avg_max_drawdown(obj_param, sec_ids=[sec_id]), 4)
            else:
                raise ValueError(obj_type + ' is not supported')

            df_list.append(pd.DataFrame(data, index=[0]))

        if 'MEAN_VARIANCE' == obj_type.upper():
            port_analytics = pd.DataFrame({'OBJECTIVE':['MEAN_VARIANCE'] * 4,
                                           'ANALYTICS': ['RETURN', 'RISK', 'RISK_TOL', '20D_AVG_MAX_DRAWDOWN'],
                                           'VALUE': [round(self.get_attr_value('RETURN'), 4),
                                                     round(self.get_risk(), 4),
                                                     round(obj_param, 4),
                                                     round(self.get_avg_max_drawdown(20), 4)]
                                           })
        elif 'MAX_SHARPE_RATIO' == obj_type.upper():
            port_analytics = pd.DataFrame({'OBJECTIVE':['MAX_SHARPE_RATIO'] * 5,
                                           'ANALYTICS': ['RETURN', 'RISK', 'SHARPE_RATIO', 'BENCHMARK_RETURN', '20D_AVG_MAX_DRAWDOWN'],
                                           'VALUE': [round(self.get_attr_value('RETURN'), 4),
                                                     round(self.get_risk(), 4),
                                                     round(self.get_sharpe_ratio(bmk=obj_param), 4),
                                                     round(obj_param, 4),
                                                     round(self.get_avg_max_drawdown(20), 4)]
                                           })
        elif 'MAX_RETURN' == obj_type.upper():
            port_analytics = pd.DataFrame({'OBJECTIVE': ['MAX_RETURN'] * 3,
                                           'ANALYTICS': ['RETURN', 'RISK', '20D_AVG_MAX_DRAWDOWN'],
                                           'VALUE': [round(self.get_attr_value('RETURN'), 4),
                                                     round(self.get_risk(), 4),
                                                     round(self.get_avg_max_drawdown(20), 4)]
                                           })
        elif 'MIN_RISK' == obj_type.upper():
            port_analytics = pd.DataFrame({'OBJECTIVE': ['MIN_RISK'] * 3,
                                           'ANALYTICS': ['RETURN', 'RISK', '20D_AVG_MAX_DRAWDOWN'],
                                           'VALUE': [round(self.get_attr_value('RETURN'), 4),
                                                     round(self.get_risk(), 4),
                                                     round(self.get_avg_max_drawdown(20), 4)]
                                           })
        elif 'RISK_PARITY' == obj_type.upper():
            port_analytics = pd.DataFrame({'OBJECTIVE': ['RISK_PARITY'] * 3,
                                           'ANALYTICS': ['RETURN', 'RISK', '20D_AVG_MAX_DRAWDOWN'],
                                           'VALUE': [round(self.get_attr_value('RETURN'), 4),
                                                     round(self.get_risk(), 4),
                                                     round(self.get_avg_max_drawdown(20), 4)]
                                           })
        elif 'MIN_AVG_MAX_DRAWDOWN' == obj_type.upper():
            port_analytics = pd.DataFrame({'OBJECTIVE': ['AVG_MAX_DRAWDOWN'] * 3,
                                           'ANALYTICS': ['RETURN', 'RISK', 'AVG_MAX_DRAWDOWN'],
                                           'VALUE': [round(self.get_attr_value('RETURN'), 4),
                                                     round(self.get_risk(), 4),
                                                     round(self.get_avg_max_drawdown(obj_param), 4)]
                                           })
        else:
            raise ValueError(obj_type + ' is not supported')

        # add additional blank line to portfolio level analytics
        port_analytics = pd.concat([port_analytics,
                                    pd.DataFrame({'OBJECTIVE': [''], 'ANALYTICS': [''], 'VALUE': ['']})
                                    ])

        return pd.concat(df_list).set_index(keys=['SEC_ID'], drop=True), port_analytics

    def reproduce_by_attr(self, attr_name, attr_values):
        """Reproduce a Portfolio object by updating its attributes.

        Args:
            attr_name (str): attribute name.
            attr_values (dict): {security id: attribute value}

        Returns:
            ptf (Portfolio): a new Portfolio object with updated attributes.
        """
        if attr_name.upper() not in self.get_attr_names():
            raise ValueError(attr_name + ' is not in the portfolio: ' + str(self.get_attr_names()))

        securities = list()
        for sec in self.get_securities():
            if sec.get_id() in attr_values:
                securities.append(sec.reproduce_by_attr(attr_name, attr_values.get(sec.get_id())))
            else:
                securities.append(sec.reproduce())

        ptf = Portfolio(self.get_weights_dict(), securities, self._covar_matrix)

        return ptf

    def reproduce_by_covar(self, covar_matrix):
        """Reproduce a Portfolio object by updating the covariance matrix.

        Args:
            covar_matrix (pandas.DataFrame): a covariance matrix in data frame, index by security IDs.

        Returns:
            ptf (Portfolio): a new Portfolio object with updated covariance matrix.
        """
        if covar_matrix is None:
            raise ValueError('Input covariance matrix is None')

        if set(covar_matrix.index) != set(covar_matrix.columns):
            raise ValueError('Index of input covar matrix is not equal to its columns.')

        covar_matrix_new = self._covar_matrix.copy(deep=True)
        for idx1 in covar_matrix.index:
            for idx2 in covar_matrix.index:
                if not math.isnan(covar_matrix.loc[idx1, idx2]):
                    covar_matrix_new.loc[idx1, idx2] = covar_matrix.loc[idx1, idx2]

        valid, err_msg = util_is_valid_covar(covar_matrix_new)
        if not valid:
            raise ValueError('Input covariance matrix is not valid: ' + err_msg)

        ptf = Portfolio(self.get_weights_dict(), self.get_securities(), covar_matrix_new)

        return ptf

    def reproduce_by_merge(self, new_sec_id, sec_ids_to_merge):
        """Reproduce a Portfolio object by merging some of its constituents.

        Args:
            new_sec_id (str): security ID of the new security formed by merging existing securities.
            sec_ids_to_merge (list): list of security IDs to merge.

        Returns:
            ptf (Portfolio): a new Portfolio object with merged constituents.
        """
        if new_sec_id is None:
            raise ValueError('Input new_sec_id is None')
        if sec_ids_to_merge is None:
            raise ValueError('Input sec_ids_to_merge is None')
        if len(sec_ids_to_merge) == 0:
            raise ValueError('Input sec_ids_to_merge is empty')

        # validate security IDs to be merged
        for sec_id in sec_ids_to_merge:
            if sec_id not in self._sec_id_list:
                raise ValueError(sec_id + ' is not in the portfolio: ' + str(self._sec_id_list))

        # build weights dictionary
        new_wt = sum([self._weights.get(id) for id in sec_ids_to_merge])
        new_wts = {id: self._weights.get(id) for id in self._sec_id_list if id not in sec_ids_to_merge}
        new_wts[new_sec_id] = new_wt

        # form a new Security object
        attributes = dict()
        for attr_nm in self.get_attr_names():
            if attr_nm == 'RISK':
                attributes[attr_nm] = self.get_risk(sec_ids_to_merge)/new_wt
            else:
                attr_value = self.get_attr_value(attr_nm, sec_ids_to_merge)
                attributes[attr_nm] = None if attr_value is None else attr_value/new_wt

        new_sec = Security(new_sec_id, attributes)
        new_secs = [self._securities.get(id) for id in self._sec_id_list if id not in sec_ids_to_merge]
        new_secs.append(new_sec)

        # compute covariance of the new security with existing securities
        covar_dict = dict() # {sec_id: covariance of sec_id and new security}
        for sec_id in self._sec_id_list: # covariance of new security with existing ones
            covar_dict[sec_id] = 0.0
            for sec_id_merge in sec_ids_to_merge:
                wt = self._weights.get(sec_id_merge)
                wtd_covar = wt * self._covar_matrix.loc[sec_id, sec_id_merge]
                covar_dict[sec_id] += wtd_covar
            covar_dict[sec_id] = covar_dict[sec_id]/new_wt
        wts_merge = np.array([self._weights.get(x) for x in self._sec_id_list if x in sec_ids_to_merge])
        covar_dict[new_sec_id] = np.matmul(np.matmul(wts_merge,
                                                     self.get_covar_matrix(sec_ids_to_merge).values),
                                           wts_merge.copy().transpose())
        covar_dict[new_sec_id] = covar_dict[new_sec_id]/(new_wt**2)

        new_id_list = [id for id in self._sec_id_list if id not in sec_ids_to_merge]
        new_id_list.append(new_sec_id)
        new_covar_matrix = pd.DataFrame(index=new_id_list, columns=new_id_list)
        for sec_id in new_id_list:
            if sec_id != new_sec_id:
                data = [self._covar_matrix.loc[idx, sec_id] for idx in self._covar_matrix.index if idx not in sec_ids_to_merge]
                data.append(covar_dict.get(sec_id))
            else:
                data = [covar_dict.get(id) for id in new_id_list]
            new_covar_matrix[sec_id] = data

        return Portfolio(weights=new_wts, securities=new_secs, covar_matrix=new_covar_matrix)

    def reproduce_by_wts(self, wts):
        """Reproduce a Portfolio object by updating the weights.

        Args:
            wts (list): an array of new weights.

        Returns:
            ptf (Portfolio): a new Portfolio object with updated weights.
        """
        if wts is None:
            raise ValueError('Input weights is None')
        if len(wts) != len(self._sec_id_list):
            raise ValueError('Input weights has wrong length')

        wts_dict = dict()
        for idx in range(len(wts)):
            wts_dict[self._sec_id_list[idx]] = wts[idx]

        ptf = Portfolio(wts_dict, self.get_securities(), self.get_covar_matrix())

        return ptf

    def to_dataframe(self, corr=True):
        """Convert a Portfolio object to a data frame.

        Returns:
            df (pandas.DataFrame): a data frame with columns 'weight', 'SEC_ID', 'attr_name1', 'attr_name2', ...
        """
        attr_nms = sorted(list(self.get_attr_names()))

        df_list = list()

        for sec_id in self._sec_id_list:
            data = dict()
            sec = self._securities.get(sec_id)

            data['SEC_ID'] = sec_id
            data['WEIGHT'] = self._weights.get(sec_id)

            # populate other attributes
            for attr_nm in attr_nms:
                data[attr_nm] = sec.get_attr_value(attr_nm)

            # populate covariance or correlation matrix
            for sec_id_for_covar in self._sec_id_list:
                if corr:
                    data[sec_id_for_covar] = self._corr_matrix.loc[sec_id, sec_id_for_covar]
                else:
                    data[sec_id_for_covar] = self._covar_matrix.loc[sec_id, sec_id_for_covar]

            df_list.append(pd.DataFrame(data, index=[0]))

        return pd.concat(df_list).set_index(keys=['SEC_ID'], drop=True)

