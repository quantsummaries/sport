# Python standard library: https://docs.python.org/3/library/index.html
import datetime
import math
import os
from typing import Dict, List

# open source packages
import pandas as pd


class CovarEstimator:
    """Estimator of covariance matrix and average returns of assets."""

    @staticmethod
    def _calc_rtrns(price: List[float], method: str) -> List[float]:
        """Calculate price returns.

        Args:
            price (list): list of prices.
            method (str): calculation method, 'logarithm' or 'simple'

        Returns:
            rtrns (list): list of returns.
        """
        if price is None:
            raise ValueError("Input price list is None")
        if not isinstance(price, list):
            raise ValueError("Input price list is not of type list")
        if len(price) == 0:
            raise ValueError("Input price list is empty")
        if method is None or not isinstance(method, str) or method not in ('simple', 'logarithm'):
            raise ValueError("Input method is not valid")

        rtrns = list()
        for idx in range(1, len(price)):
            if method == 'simple':
                rtrn = price[idx]/price[idx-1] - 1
            elif method == 'logarithm':
                rtrn = math.log(price[idx]) - math.log(price[idx-1])
            else:
                raise ValueError('Input method is invalid: must be log or simple')
            rtrns.append(rtrn)

        return rtrns

    @staticmethod
    def _preprocess_data(data_dir: str, price_type: str) -> (Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]):
        """Collect price data and other data.

        Args:
            data_dir (str): a directory which holds all the security price data.
            price_type (str): type of price used for return calculation, e.g. Close, Adjust Close, etc.
        Returns:
            sec_id_to_price (dict): {sec_id: pandas.DataFrame (columns=['DATE', 'PRICE'])}.
            sec_id_to_last_data (dict): {sec_id: dictionary of data on the last date}.
        """
        if data_dir is None:
            raise ValueError('Input data directory is None')
        if not isinstance(data_dir, str):
            raise ValueError(f"""Input data directory is not of type str: {data_dir}""")
        if not os.path.exists(data_dir):
            raise ValueError(f"""Input data directory does not exist: {data_dir}""")

        if price_type is None:
            raise ValueError('Input price type is None')
        if not isinstance(price_type, str):
            raise ValueError(f"""Input price type is not of type str: {price_type}""")

        sec_id_to_price = dict()
        sec_id_to_last_data = dict()
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if "csv" not in file:
                    continue
                data = pd.read_csv(os.path.join(root, file),
                                   dtype={'DATE': 'str',
                                          'SEC_ID': 'str',
                                          'SEC_NM': 'str',
                                          'OPEN': 'float',
                                          'HIGH': 'float',
                                          'LOW': 'float',
                                          'CLOSE': 'float',
                                          'VOLUME': 'int',
                                          'NUM_SHARES': 'float',
                                          'AMOUNT': 'float',
                                          'NUM_TRANSACTIONS': 'int'})

                if data.shape[0] < 2:
                    print(file + ' has less than 2 data points. Skipped.')
                    continue

                # choose 'CLOSE' as 'PRICE'
                if price_type not in data.columns:
                    raise ValueError(f"""price data in {file} does not have a column {price_type} that can be used for return calculation.""")
                price_data = data[['DATE', price_type]].copy()
                price_data.columns = ['DATE', 'PRICE']
                price_data['DATE'] = [datetime.datetime.strptime(x, "%Y-%m-%d").date() for x in price_data['DATE']]

                # other useful data on the last date: sec_id, sec_nm, any analytics, etc.
                other_data = data.iloc[-1].to_dict()

                sec_id = other_data.get('SEC_ID')

                sec_id_to_last_data[sec_id] = other_data

                # if there is no NA in 'PRICE', store the data and continue
                if not price_data['PRICE'].isna().values.any():
                    sec_id_to_price[sec_id] = price_data
                    continue

                print(f"""WARNING: price data in {file} have NA; use previous valid price to impute ...""")

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

    def __init__(self, data_dir: str,
                 rtrn_method: str,
                 halflife_in_yrs: float,
                 cash_rtrn: float,
                 price_type: str,
                 sec_id_list: List[str] = None,
                 start_dt: datetime.date = None,
                 end_dt: datetime.date = None) -> None:
        """
        Args:
            data_dir (str): a directory which holds all the security price data.
            rtrn_method (str): 'simple' or 'logarithm' for return calculation.
            halflife_in_yrs (float): halflife for the decay factor in covariance weighting scheme.
            cash_rtrn (float): risk free cash rate.
            price_type (str): type of price used for return calculation, e.g. Close, Adjust Close, etc.
            sec_id_list (List[str]): list of securities used to covar estimation.
            start_dt (datetime.date): start date of estimation window.
            end_dt (datetime.date); end date of estimation window.
        """
        if data_dir is None:
            raise ValueError('Input data directory is None')
        if not isinstance(data_dir, str):
            raise ValueError(f"""Input data directory is not of type str: {data_dir}""")
        if not os.path.exists(data_dir):
            raise ValueError(f"""Input data directory does not exist: {data_dir}""")

        if rtrn_method is None or not isinstance(rtrn_method, str) or rtrn_method not in ('simple', 'logarithm'):
            raise ValueError("Input method is not valid")

        if halflife_in_yrs is None or not isinstance(halflife_in_yrs, float):
            raise ValueError("Input halflife_in_yrs is not valid")

        if cash_rtrn is None or not isinstance(cash_rtrn, float):
            raise ValueError("Input cash_rtrn is not valid")

        if price_type is None:
            raise ValueError('Input price type is None')
        if not isinstance(price_type, str):
            raise ValueError(f"""Input price type is not of type str: {price_type}""")

        if sec_id_list is not None:
            if not isinstance(sec_id_list, list):
                raise ValueError(f"""Input sec_id_list is not of type list""")
            if len(sec_id_list) == 0:
                raise ValueError("Input sec_id_list is empty")

        if start_dt is not None and end_dt is not None:
            if not isinstance(start_dt, datetime.date):
                raise ValueError(f"""Input start_dt is not of type datetime.date: {start_dt}""")
            if not isinstance(end_dt, datetime.date):
                raise ValueError(f"""Input end_dt is not of type datetime.date: {end_dt}""")
            if start_dt >= end_dt:
                raise ValueError(f"""start_dt {start_dt} is not before end_dt {end_dt}""")

        self._sec_id_list = None
        self._sec_id_to_last_data = None
        self._avg_rtrns = dict()
        self._covar_matrix = None
        self._risks = dict()

        self._sec_id_to_last_data, sec_id_to_price = CovarEstimator._preprocess_data(data_dir=data_dir, price_type=price_type)

        # remove sec_id that are not needed
        if sec_id_list is not None:
            # check all sec_id's in input list have data
            for x in sec_id_list:
                if x not in sec_id_to_price:
                    raise ValueError(f"""data for sec_id {x} are not in {data_dir}""")
            # remove sec_id's that are not needed
            for x in list(sec_id_to_price.keys()):
                if x not in sec_id_list:
                    self._sec_id_to_last_data.pop(x)
                    sec_id_to_price.pop(x)

        # remove data that are outside [start_dt, end_dt]
        if start_dt is not None and end_dt is not None:
            for sec_id, data in sec_id_to_price.items():
                if data['DATE'].iloc[0] > start_dt or data['DATE'].iloc[-1] < end_dt:
                    raise ValueError(f"""price data for {sec_id} cover [{data['DATE'].iloc[0]}, {data['DATE'].iloc[-1]}], but not the asked time window [{start_dt}, {end_dt}]""")
                data = data[(data['DATE'] >= start_dt) & (data['DATE'] <= end_dt)]

        self._sec_id_list = sorted([x for x in self._sec_id_to_last_data])

        print(f"""Estimating covariance for {self._sec_id_list} by {rtrn_method} returns: {start_dt} - {end_dt}""")

        rtrns_dict = dict()
        for sec_id in sec_id_to_price:
            # calculate and store return time series for each security
            rtrn = CovarEstimator._calc_rtrns(price=sec_id_to_price.get(sec_id)['PRICE'].to_list(),
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

        # generate correlation matrix based on covar matrix
        self._corr_matrix = self._covar_matrix.copy()
        stdvar = dict()
        for idx in self._covar_matrix.index:
            stdvar[idx] = math.sqrt(self._covar_matrix.loc[idx, idx])
        for idx in self._covar_matrix.index:
            for col in self._covar_matrix.columns:
                self._corr_matrix.loc[idx, col] = round(self._corr_matrix.loc[idx, col]/(stdvar[idx]*stdvar[col]),6)

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
        self._sec_id_to_last_data['000000'] = {'SEC_NM': 'CASH'}
        self._avg_rtrns['000000'] = cash_rtrn
        self._risks['000000'] = 0.0

        new_columns = list(self._covar_matrix.columns)
        new_columns.append('000000')
        cash_dataframe = pd.DataFrame(index=['000000'], columns=new_columns)
        for col in cash_dataframe.columns:
            cash_dataframe.loc['000000', col] = 0.0
        self._covar_matrix = pd.concat([self._covar_matrix, cash_dataframe])
        self._covar_matrix['000000'] = [0.0] * self._covar_matrix.shape[0]

    def get_avg_rtrns(self) -> Dict:
        """Return a dictionary of average resturns.

        Returns:
            avg_rtrns (dict): {sec_id: avg_rtrn}.
        """
        return self._avg_rtrns.copy()

    def get_covar_matrix(self) -> pd.DataFrame:
        """Return a data frame of covar matrix.

        Returns:
            covar_matrix (pandas.DataFrame): a data frame of covariance matrix with index and columns being IDs.
        """
        return self._covar_matrix.copy()

    def get_corr_matrix(self) -> pd.DataFrame:
        """Return a data frame of correlation matrix.

        Returns:
            corr_matrix (pandas.DataFrame): a data frame of correlation matrix with index and columns being IDs.
        """
        return self._corr_matrix.copy()

    def get_risk(self) -> Dict[str, float]:
        """Return a dictionary of risk.

        Returns:
            risk (dict): {sec_id: risk}.
        """
        return self._risks.copy()

    def to_dataframe(self) -> pd.DataFrame:
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
