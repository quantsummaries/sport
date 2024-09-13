import os
import traceback

import numpy as np
import pandas as pd

from sport import Dao


def test():
    """Unit tests."""

    dir_path = os.path.abspath(os.path.dirname(__file__))

    covar_file_path = os.path.join(dir_path, 'test_data', 'covar_matrix.csv')
    covar = pd.read_csv(covar_file_path, dtype={'SEC_ID': 'str'})
    covar.set_index('SEC_ID', inplace=True, drop=True)

    attr_file_path = os.path.join(dir_path, 'test_data', 'attributes_data.csv')
    attr = pd.read_csv(attr_file_path, dtype={'SEC_ID': 'str', 'IND_CD': 'str'})
    attr.set_index('SEC_ID', inplace=True, drop=True)

    dao = Dao.init_from_dataframes(covar=covar, attributes=attr)
    secs = dao.get_securities_list(sec_id_list=['600276', '300347'])
    assert '801150' == secs[0].get_attr_value('IND_CD')
    assert np.isclose(0.09972711780656822, secs[1].get_attr_value('DIV_YLD'))

    print(dao.get_covar_matrix())
    print(dao.to_dataframe())


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)
        test()
    except Exception as err:
        print("Dao unit port failed: ", err)
        print(traceback.format_exc())
