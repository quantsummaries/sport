import math
import os

import pandas as pd


class CovarEstimator:
    """Estimator of covariance matrix and average returns of assets."""

    @staticmethod
    def _calc_rtrns(price, method):
        """Calculate price returns.

        Args:
            price (list): list of prices.
            method (str): calculation method, 'logarithm' or 'simple'

        Returns:
            rtrns (list): list of returns.
        """
        rtrns = list()
        for idx in range(len(price)):
            if idx == 0:
                continue
            else:
                if method == 'simple':
                    rtrn = price[idx]/price[idx-1] - 1
                elif method == 'logarithm':
                    rtrn = math.log(price[idx]) - math.log(price[idx-1])
                else:
                    raise ValueError('Input method is invalid: must be log or simple')
            rtrns.append(rtrn)

        return rtrns

    @staticmethod
    def _preprocess_data(data_dir):
        """Collect price data and other data.

        Args:
            data_dir (str): a directory which holds all the security price data.

        Returns:
            sec_id_to_price (dict): {sec_id: pandas.DataFrame (columns=['DATE', 'PRICE'])}.
            sec_id_to_last_data (dict): {sec_id: dictionary of data on the last date}.
        """
        if data_dir is None:
            raise ValueError('Input data directory is None')

        sec_id_to_price = dict()
        sec_id_to_last_data = dict()
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                data = pd.read_csv(os.path.join(root, file),
                                   dtype={'DATE': 'str',
                                          'SEC_ID': 'str',
                                          'SEC_NM': 'str',
                                          'OPEN': 'float',
                                          'HIGH': 'float',
                                          'LOW': 'float',
                                          'CLOSE': 'float',
                                          'NUM_SHARES': 'float',
                                          'AMOUNT': 'float',
                                          'NUM_TRANSACTIONS': 'int'})

                if data.shape[0] < 2:
                    print(file + ' has less than 2 data points. Skipped.')
                    continue

                # choose 'CLOSE' as 'PRICE'
                price_data = data[['DATE', 'CLOSE']]
                price_data.columns = ['DATE', 'PRICE']

                # other useful data on the last date: sec_id, sec_nm, any analytics, etc.
                other_data = data.iloc[-1].to_dict()

                sec_id = other_data.get('SEC_ID')

                sec_id_to_last_data[sec_id] = other_data

                # if there is no NA in 'PRICE', store the data and continue
                if not price_data['PRICE'].isna().values.any():
                    sec_id_to_price[sec_id] = price_data
                    continue

                # if there is NA in 'PRICE', impute using previous 'PRICE'
                print(sec_id + ' has NONE in CLOSE')
                bad_idx = set()
                for idx in price_data.index:
                    if price_data.loc[idx, 'PRICE'] is None:
                        if (idx > 0) and (not price_data.loc[idx - 1, 'PRICE'] is None):
                            price_data.loc[idx, 'PRICE'] = data.loc[idx - 1, 'PRICE']
                        else:
                            bad_idx.add(idx)

                price_data = price_data[~price_data.index.isin(bad_idx)]
                sec_id_to_price[sec_id] = price_data

        return sec_id_to_last_data, sec_id_to_price

    def __init__(self, data_dir, rtrn_method, halflife_in_yrs):
        """
        Args:
            data_dir (str): a directory which holds all the security price data.
            rtrn_method (str): 'simple' or 'logarithm' for return calculation.
            halflife_in_yrs (float): halflife for the decay factor in covariance weighting scheme.
        """
        self._sec_id_list = None
        self._sec_id_to_last_data = None
        self._avg_rtrns = dict()
        self._covar_matrix = None
        self._risks = dict()

        self._sec_id_to_last_data, sec_id_to_price = CovarEstimator._preprocess_data(data_dir)
        self._sec_id_list = sorted([x for x in self._sec_id_to_last_data])

        rtrns_dict = dict()
        for sec_id in sec_id_to_price:
            # calculate and store return time series for each security
            rtrn = CovarEstimator._calc_rtrns(price=sec_id_to_price.get(sec_id)['PRICE'],
                                              method=rtrn_method)
            dates = sec_id_to_price.get(sec_id)['DATE'][1:]
            rtrns_dict[sec_id] = pd.DataFrame.from_dict({'DATE': dates, 'RETURN': rtrn}).set_index(keys=['DATE'],
                                                                                                   drop=True)

            # estimated average return
            self._avg_rtrns[sec_id] = sum(rtrn)/len(rtrn)

        # calculate covariance matrix
        if halflife_in_yrs > 0:
            decay_factor = 0.5 ** (1/(250.0 * halflife_in_yrs))
        else:
            decay_factor = 1

        self._covar_matrix = pd.DataFrame(index=self._sec_id_list, columns=self._sec_id_list)
        for idx1 in range(len(self._sec_id_list)):
            for idx2 in range(len(self._sec_id_list)):
                if idx1 > idx2:
                    continue
                sec_id1 = self._sec_id_list[idx1]
                sec_id2 = self._sec_id_list[idx2]
                rtrns1 = rtrns_dict.get(sec_id1)
                rtrns2 = rtrns_dict.get(sec_id2)

                comm_idx = rtrns1.index.intersection(rtrns2.index)
                rtrns1 = list(rtrns1[rtrns1.index.isin(comm_idx)]['RETURN'])
                rtrns2 = list(rtrns2[rtrns2.index.isin(comm_idx)]['RETURN'])

                wts = [decay_factor ** n for n in range(len(rtrns1))]
                wts.reverse()
                total_wt = sum(wts)
                wts = [x/total_wt for x in wts]

                avg1 = self._avg_rtrns.get(sec_id1)
                avg2 = self._avg_rtrns.get(sec_id2)
                covariance = list()
                for idx in range(len(rtrns1)):
                    covariance.append(wts[idx] * (rtrns1[idx] - avg1) * (rtrns2[idx] - avg2))
                self._covar_matrix.loc[sec_id1, sec_id2] = sum(covariance)
                self._covar_matrix.loc[sec_id2, sec_id1] = self._covar_matrix.loc[sec_id1, sec_id2]

        # re-scale daily return to annualized return
        rtrn_scalar = 243
        for k in self._avg_rtrns:
            self._avg_rtrns[k] = rtrn_scalar * self._avg_rtrns.get(k)

        # re-scale daily volatility to annualized volatility
        for sec_id1 in self._sec_id_list:
            for sec_id2 in self._sec_id_list:
                self._covar_matrix.loc[sec_id1, sec_id2] *= rtrn_scalar

        # populate risk
        for sec_id in self._sec_id_list:
            self._risks[sec_id] = math.sqrt(self._covar_matrix.loc[sec_id, sec_id])

        # add CASH as a stand-alone asset
        self._sec_id_list.append('000000')
        self._sec_id_to_last_data['000000'] = {'SEC_NM': '现金'}
        self._avg_rtrns['000000'] = 0.0035
        self._risks['000000'] = 0.0

        new_columns = list(self._covar_matrix.columns)
        new_columns.append('000000')
        cash_dataframe = pd.DataFrame(index=['000000'], columns=new_columns)
        for col in cash_dataframe.columns:
            cash_dataframe.loc['000000', col] = 0.0
        self._covar_matrix = pd.concat([self._covar_matrix, cash_dataframe])
        self._covar_matrix['000000'] = [0.0] * self._covar_matrix.shape[0]

    def get_avg_rtrns(self):
        """Return a dictionary of average resturns.

        Returns:
            avg_rtrns (dict): {sec_id: avg_rtrn}.
        """
        return self._avg_rtrns.copy()

    def get_covar_matrix(self):
        """Return a data frame of covar matrix.

        Returns:
            covar_matrix (pandas.DataFrame): a data frame of covariance matrix with index and columns being IDs.
        """
        return self._covar_matrix.copy()

    def get_risk(self):
        """Return a dictionary of risk.

        Returns:
            risk (dict): {sec_id: risk}.
        """
        return self._risks.copy()

    def to_dataframe(self):
        """Return a data frame of return, risk, and covariance matrix, with sec_id as the index.

        Returns:
            df (pandas.DataFrame): data frame with columns 'RETURN', 'RISK', 'sec_id1', 'sec_id2', ...
        """
        dict_for_df = dict()
        dict_for_df['SEC_ID'] = self._sec_id_list

        # add additional information
        dict_for_df['SEC_NM'] = [self._sec_id_to_last_data.get(sec_id).get('SEC_NM') for sec_id in self._sec_id_list]
        dict_for_df['LAST_PRICE_DATE'] = [self._sec_id_to_last_data.get(sec_id).get('DATE') for sec_id in self._sec_id_list]

        dict_for_df['RETURN'] = [self._avg_rtrns.get(sec_id) for sec_id in self._sec_id_list]
        dict_for_df['RISK'] = [self._risks.get(sec_id) for sec_id in self._sec_id_list]
        for sec_id in self._sec_id_list:
            dict_for_df[sec_id] = list(self._covar_matrix[sec_id])

        df = pd.DataFrame.from_dict(data=dict_for_df)
        df.set_index('SEC_ID', inplace=True)

        return df
