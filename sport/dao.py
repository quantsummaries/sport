import math
import os
import warnings

import pandas as pd

from sport import Constants, Security

class Dao:
    """Data access object class to get security attributes."""

    @staticmethod
    def read_excel_input(file_path):
        """Read inputs from an Excel file with given format.

        Args:
            file_path (string): file path of the Excel input file.

        Returns:
            sec_id_list (list): list of security IDs.
            returns (dict): {security id: nonp-NaN return rate}.
            covar (pandas.DataFrame): covariance matrix in data frame format.
            objectives (dict): {'MEAN_VARIANCE': risk_tol, 'MAX_SHARPE_RATIO': benchmark rate, others: NaN}.
            constr_df (pandas.DataFrame): a data frame of constraints.
        """

        # '投资组合'
        input_port = pd.read_excel(io=file_path, sheet_name='投资组合', dtype={'证券代码': 'str', '证券名称': 'str', '年化收益率': 'float'})
        sec_id_list = input_port['证券代码'].to_list()
        returns = {k: v for k, v in zip(sec_id_list, input_port['预期年化收益率'].to_list()) if not math.isnan(v)}

        # '优化目标'
        input_objective = pd.read_excel(io=file_path, sheet_name='优化目标',
                                        dtype={'优化目标': 'str', '选择该优化目标': 'str', '参数': 'float'})
        objectives = dict()
        for idx in input_objective.index:
            if input_objective.loc[idx, '选择该优化目标'] == '是':
                obj_cn = input_objective.loc[idx, '优化目标']
                objectives[Constants.cn_to_en.get(obj_cn)] = input_objective.loc[idx, '参数']

        # '限制条件'
        input_constr = pd.read_excel(io=file_path, sheet_name='限制条件',
                                     dtype={'投资标的': 'str', '限制属性': 'str', '限制属性参数': 'float', '最小值': 'float', '最大值': 'float'})
        input_constr.columns = [Constants.cn_to_en.get(col) for col in input_constr.columns]
        for idx in input_constr.index:
            if input_constr.loc[idx, 'SEC_ID'] in {'每个', '全部'}:
                input_constr.loc[idx, 'SEC_ID'] = Constants.cn_to_en.get(input_constr.loc[idx, 'SEC_ID'])
            val = Constants.cn_to_en.get(input_constr.loc[idx, 'ATTRIBUTE'])
            if val is None:
                raise ValueError(input_constr.loc[idx, 'ATTRIBUTE'] + ' is not supported')
            else:
                input_constr.loc[idx, 'ATTRIBUTE'] = val

        constr_df = pd.DataFrame(columns=['TYPE', 'SEC_ID', 'ATTRIBUTE', 'ATTRIBUTE_PARAMS', 'VALUE'])
        for idx in input_constr.index:
            sec_id = input_constr.loc[idx, 'SEC_ID']
            attribute = input_constr.loc[idx, 'ATTRIBUTE']
            attr_params = input_constr.loc[idx, 'ATTRIBUTE_PARAMS']
            min = input_constr.loc[idx, 'MIN']
            max = input_constr.loc[idx, 'MAX']
            if attribute in Constants.nonlinear_constr:
                type = 'NONLINEAR_'
            else:
                type = 'LINEAR_'
            if math.isclose(min, max):
                type = type + 'EQ'
                value = str(min)
            elif min < max:
                type = type + 'INEQ'
                value = '[' + str(min) + ':' + str(max) + ']'
            else:
                raise ValueError('min is greater than max')
            constr_df.loc[idx, 'TYPE'] = type
            constr_df.loc[idx, 'SEC_ID'] = sec_id
            constr_df.loc[idx, 'ATTRIBUTE'] = attribute
            constr_df.loc[idx, 'ATTRIBUTE_PARAMS'] = attr_params
            constr_df.loc[idx, 'VALUE'] = value

        constr_df.loc[idx+1, 'TYPE'] = 'LINEAR_EQ'
        constr_df.loc[idx+1, 'SEC_ID'] = 'ALL'
        constr_df.loc[idx+1, 'ATTRIBUTE'] = 'WEIGHT'
        constr_df.loc[idx+1, 'ATTRIBUTE_PARAMS'] = None
        constr_df.loc[idx+1, 'VALUE'] = 1

        # '客户的协方差矩阵'
        covar = pd.read_excel(io=file_path, sheet_name='客户的协方差矩阵', dtype={'证券代码': 'str'})
        covar.columns = ['SEC_ID' if x == '证券代码' else x for x in covar.columns]
        covar.set_index(keys=['SEC_ID'], drop=True, inplace=True)

        # '客户的投资基准'
        #benchmark = pd.read_excel(io=file_path, sheet_name='客户的投资基准', dtype={'投资基准代码': 'str'})
        #benchmark = benchmark[['日期', '日收益率']]
        #benchmark.columns = ['DATE', 'RETURN']
        #print(benchmark)
        return sec_id_list, returns, covar, objectives, constr_df

    @staticmethod
    def write_to_excel(file_path, dfs):
        """Write to an Excel file a dictionary of data frames.

        Args:
            file_path (str): path of the output file.
            dfs (dict): dictionary of data frames {worksheet name: data frame}
        """
        writer = pd.ExcelWriter(file_path, engine='openpyxl')

        for worksheet_nm in dfs:
            df = dfs.get(worksheet_nm)
            df.to_excel(writer, sheet_name=worksheet_nm, index=False)

        # writer.save()
        writer.close()

    @classmethod
    def init_from_dataframes(cls, covar, attributes):
        """Factory method to construct a Dao object from data frames.

        Args:
             covar (pandas.DataFrame): a data frame that contains covariance matrix.
             attributes (pandas.DataFrame): a data frame that contains security attributes information.
        """
        if covar is None:
            raise ValueError('Covar matrix data frame is None')
        if attributes is None:
            raise ValueError('Security attributes data frame is None')

        return Dao({'Covar': covar, 'Attributes': attributes})

    @classmethod
    def init_from_default_data(cls, covar_path: str, attributes_path: str):
        """Factory method to construct a Dao object from default data."""

        # load covar matrix
        if not os.path.exists(covar_path):
            raise ValueError(covar_path + ' does not exist; covar matrix data frame is set to None')
        else:
            covar = pd.read_csv(covar_path, dtype={'SEC_ID': 'str'}).set_index(keys=['SEC_ID'], drop=True)

        # load attributes data
        if not os.path.exists(attributes_path):
            raise ValueError(attributes_path + ' does not exist; security attributes data frame is set to None')
        else:
            attributes = pd.read_csv(attributes_path, dtype={'SEC_ID': 'str', 'IND_CD': 'str'}).set_index(
                keys=['SEC_ID'], drop=True)

        return Dao.init_from_dataframes(covar, attributes)

    def __init__(self, args):
        """
        Args dictionary keys: {'Covar', 'Attributes'}

        Args:
             args (dict): {'Covar': covariance matrix, 'Attributes': a data frame of security attributes}.
        """
        if args is None:
            raise ValueError('Input args is None')
        if len(args) == 0:
            raise ValueError('Input args is empty')

        self._securities = list()   # list of objects of Security class.
        self._id_list = None        # list of string
        self._id_set = None         # set of string
        self._attr_set = None       # set of string
        self._covar_matrix = None   # data frame

        if set(args.keys()) != {'Covar', 'Attributes'}:
            raise ValueError('Input set ' + str(args.keys()) + ' are not supported')

        covar = args.get('Covar')
        attributes = args.get('Attributes')

        # identify securities which do not have relevant attributes.
        if not set(covar.index).issubset(set(attributes.index)):
            for idx in covar.index:
                if idx not in set(attributes.index):
                    warnings.warn(idx + ' is in covar data frame but not in attributes data frame')

        if not set(covar.index).issubset(covar.columns):
            raise ValueError('Covar data frame row index is not a subset of columns')

        # self._id_list based on covar matrix
        self._id_list = list(covar.index)
        self._id_set = set(covar.index)

        # self._covar_matrix
        self._covar_matrix = covar.loc[self._id_list, self._id_list]

        # self._attr_set: SEC_NM, RETURN, RISK from covar matrix; others from attributes
        self._attr_set = (set(covar.columns) - set(covar.index)).union(attributes.columns)

        # self._securities
        for sec_id in self._id_list:
            sec_attributes = dict()
            for attr_nm in self._attr_set:
                if attr_nm in covar.columns:
                    sec_attributes[attr_nm] = covar.loc[sec_id, attr_nm]
                elif attr_nm in attributes.columns and sec_id in attributes.index:
                    sec_attributes[attr_nm] = attributes.loc[sec_id, attr_nm]
                else:
                    sec_attributes[attr_nm] = None

            self._securities.append(Security(sec_id, sec_attributes))

    def get_corr_matrix(self):
        """Get the correlation matrix of the security returns.

        Returns:
            corr_matrix (pandas.DataFrame): correlation matrix with IDs as columns and index.
        """
        return self._corr_matrix.copy(deep=True)

    def get_covar_matrix(self, sec_id_list=None):
        """Get the covariance matrix of the security returns.

        Args:
            sec_id_list (list): list of security IDs by which the covariance matrix will be fetched.
        Returns:
            covar_matrix (pandas.DataFrame): covariance matrix with IDs as columns and index.
        """
        if sec_id_list is None:
            return self._covar_matrix.copy(deep=True)
        else:
            for idx in sec_id_list:
                if idx not in self._id_list:
                    raise ValueError(idx + ' is not in the security universe')

            return self._covar_matrix.loc[sec_id_list, sec_id_list].copy(deep=True)

    def get_securities_list(self, sec_id_list=None):
        """Get a list of Security objects.

        Args:
            sec_id_list (list): list of security IDs by which the securities will be fetched.

        Returns:
            result (list): a list of Security objects.
        """
        if sec_id_list is None:
            return self._securities.copy()
        else:
            secs = self.get_securities_dict(sec_id_list)
            return [secs.get(sec_id) for sec_id in sec_id_list]

    def get_securities_dict(self, sec_id_list=None):
        """Get a dictionary of Security objects: {security id: Security object}.

        Args:
            sec_id_list (list): list of security IDs by which the securities will be fetched.

        Returns:
            result (dict): {security id: Security object}.
        """
        if sec_id_list is None:
            return {x.get_id(): x for x in self._securities}
        else:
            return {x.get_id(): x for x in self._securities if x.get_id() in sec_id_list}

    def to_dataframe(self):
        """Get the data in the format of a data frame.

        Returns:
            df (pandas.DataFrame): fetched data in a data frame.
        """
        df_list = list()
        for sec in self.get_securities_list():
            data_dict = {'SEC_ID': [sec.get_id()]}

            for attr_nm in sorted(list(sec.get_attr_names())):
                data_dict[attr_nm] = [sec.get_attr_value(attr_nm)]

            for sec_id in self._id_list:
                data_dict[sec_id] = [self._covar_matrix.loc[sec.get_id(), sec_id]]

            df_list.append(pd.DataFrame.from_dict(data_dict))

        df = pd.concat(df_list).set_index(keys=['SEC_ID'], drop=True)

        return df
