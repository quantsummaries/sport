import math
import os
import traceback

import numpy as np
import pandas as pd

from sport import Dao, Portfolio, Security


def test():
    """Unit tests."""

    # __init__
    sec1 = Security('1234', {'duration': 7.0, 'oas': 200.0})
    sec2 = Security('5678', {'ytm': 0.3, 'oas': 300.0})
    covar = pd.DataFrame(data={'1234': [1.0, 0.5], '5678': [0.5, 0.7]},
                         index=['1234', '5678'],
                         dtype=float)

    try:
        port = Portfolio({'1234': 0.5, '456': 0.5}, [sec1, sec2], covar)
    except ValueError as err:
        assert str(err) == "Security ID 456 is in the weight dictionary but not in the security list"

    try:
        port = Portfolio({'1234': 0.5, '5678': 0.2}, [sec1, sec2], covar)
    except ValueError as err:
        assert str(err) == 'Sum of weights 0.7 != 1.0'

    port = Portfolio(weights={'1234': 0.7, '5678': 0.3},
                     securities=[sec1, sec2],
                     covar_matrix=covar)

    # get_attr_names()
    assert {'DURATION', 'YTM', 'OAS'} == port.get_attr_names()

    # get_attr_values(...)
    assert math.isclose(port.get_attr_value('oas'), 230.0)
    assert math.isclose(port.get_attr_value('duration'), 4.9)
    assert port.get_attr_value('abcd') is None

    # get_contr_to_risk(...)
    assert math.isclose(0.7 * 0.85 / 0.8734987120768982, port.get_contr_to_risk('1234'))

    # get_covar_matrix(...)
    assert math.isclose(1.0, port.get_covar_matrix(['1234']).loc['1234', '1234'])
    assert math.isclose(0.5, port.get_covar_matrix(['5678', '1234']).loc['5678', '1234'])
    try:
        assert math.isclose(0.5, port.get_covar_matrix(['abcd', '1234']).loc['abcd', '1234'])
    except ValueError as err:
        assert str(err) == 'abcd' + ' is not in the portfolio'

    # get_risk(...)
    print(port.to_dataframe())
    assert math.isclose(0.7, port.get_risk(['1234']))
    assert math.isclose(0.3 * np.sqrt(0.7), port.get_risk(['5678']))
    assert math.isclose(np.sqrt(np.matmul(np.matmul(np.array([0.7, 0.3]), port.get_covar_matrix().to_numpy()),
                                np.array([0.7, 0.3]))
                                ),
                        port.get_risk())

    # get_security(...)
    print(port.get_security('1234').to_dataframe())

    # get_securities()
    secs = port.get_securities()
    assert len(secs) == 2
    assert secs[0].get_id() == '1234'
    assert secs[1].get_id() == '5678'

    # get_weights_list(), get_weights_dict()
    assert port.get_weights_list(['1234'])[0] == 0.7
    assert port.get_weights_list(['5678'])[0] == 0.3
    assert port.get_weights_dict().get('5678') == 0.3

    # to_dataframe(), reproduce_by_{attr, wts, covar}(...)
    assert np.allclose(port.reproduce_by_attr('duration', {'5678': 15.0}).to_dataframe()['DURATION'],
                             [7.0, 15.0])
    assert np.allclose(port.reproduce_by_wts([0.5, 0.5]).to_dataframe()['WEIGHT'], [0.5, 0.5])
    new_covar_matrix = port.get_covar_matrix()+1
    new_port = port.reproduce_by_covar(new_covar_matrix)
    assert np.allclose(new_port.get_covar_matrix().to_numpy(),
                       port.get_covar_matrix().to_numpy() + 1)

    dir_path = os.path.abspath(os.path.dirname(__file__))

    covar_file_path = os.path.join(dir_path, 'test_data', 'covar_matrix.csv')
    covar = pd.read_csv(covar_file_path, dtype={'SEC_ID': 'str'}).set_index(keys=['SEC_ID'], drop=True)

    attr_file_path = os.path.join(dir_path, 'test_data', 'attributes_data.csv')
    attr = pd.read_csv(attr_file_path, dtype={'SEC_ID': 'str', 'IND_CD': 'str'}).set_index(keys=['SEC_ID'], drop=True)

    dao = Dao.init_from_dataframes(covar, attr)

    sec_ids = ['002138', '600030', '600276']
    weights = {sec_id: 1.0/len(sec_ids) for sec_id in sec_ids}

    port = Portfolio(weights, dao.get_securities_list(sec_ids), dao.get_covar_matrix(sec_ids))
    assert port.get_attr_value('NAME') is None

    # reproduce_by_attr(...)
    new_port = port.reproduce_by_attr('RETURN', {'002138': 0.999})
    assert np.isclose(new_port.get_security('002138').get_attr_value('RETURN'), 0.999)

    # reproduce_by_merge(...)
    merged_ptf = port.reproduce_by_merge('600xxx', ['600030', '600276'])
    for attr_nm in {'RETURN', 'DIV_YLD'}:
        bmk = port.get_attr_value(attr_nm, ['600030', '600276'])
        merged = merged_ptf.get_attr_value(attr_nm, ['600xxx'])
        assert math.isclose(bmk, merged)
    bmk = merged_ptf.get_security('600xxx').get_attr_value('risk') # calculated from original portfolio
    merged = merged_ptf.get_risk(['600xxx'])/(weights['600030']+weights['600276']) # calculated from new covar matrix
    assert math.isclose(bmk, merged)


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)
        test()
    except Exception as err:
        print("Portfolio unit port failed: " + str(err))
        print(traceback.format_exc())