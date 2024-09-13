import math
import os
import traceback

from cvxopt import matrix
import numpy as np
import pandas as pd

from sport import Dao, Objective, Portfolio
from sport.functions import obj_qp, obj_neg_sharpe_ratio


def test():
    test_objective_generation()
    test_params_generation()


def test_objective_generation():
    Q = np.array([[1.0, 3.0], [2.0, 4.0]])
    p = np.array([1.0, 1.0])
    x = np.array([7.0, 8.0])
    assert math.isclose(600.0, obj_qp(x, {'Q': Q, 'p': p}))
    assert math.isclose(-0.5 / np.sqrt(6.75), obj_neg_sharpe_ratio([0.5, 1], {'Q': Q, 'p': p, 'bmk': 1}), abs_tol=1e-5)

    # get_obj_cvxopt_solvers_qp() via direct inputs
    obj_direct = Objective({'Q': matrix(Q), 'p': matrix(p)})
    obj_cvxopt_direct = obj_direct.get_obj_cvxopt_solvers_qp()

    print(obj_cvxopt_direct.get('Function'))
    print(obj_cvxopt_direct.get('Params'))

    # get_obj_cvxopt_solvers_qp() via portfolio
    dir_path = os.path.abspath(os.path.dirname(__file__))

    covar_file_path = os.path.join(dir_path, 'test_data', 'covar_matrix.csv')
    covar = pd.read_csv(covar_file_path, dtype={'SEC_ID': 'str'}).set_index(keys=['SEC_ID'], drop=True)

    attr_file_path = os.path.join(dir_path, 'test_data', 'attributes_data.csv')
    attr = pd.read_csv(attr_file_path, dtype={'SEC_ID': 'str', 'IND_CD': 'str'}).set_index(keys=['SEC_ID'], drop=True)

    dao = Dao.init_from_dataframes(covar, attr)

    weights = {sec_id: 1.0 / len(dao.get_securities_list()) for sec_id in dao.get_securities_dict()}
    port = Portfolio(weights, dao.get_securities_list(), dao.get_covar_matrix())

    obj_ptf = Objective({'Function': obj_qp, 'Params': Objective.produce_params_qp(ptf=port, risk_tol=1)})
    obj_cvxopt_ptf = obj_ptf.get_obj_cvxopt_solvers_qp()
    obj_scipy_ptf = obj_ptf.get_obj_scipy_optimize_minimize_trust_constr()

    print(obj_cvxopt_ptf)
    print(obj_scipy_ptf)


def test_params_generation():
    dir_path = os.path.abspath(os.path.dirname(__file__))

    covar_file_path = os.path.join(dir_path, 'test_data', 'covar_matrix.csv')
    covar = pd.read_csv(covar_file_path, dtype={'SEC_ID': 'str'})
    covar.set_index('SEC_ID', inplace=True, drop=True)

    attr_file_path = os.path.join(dir_path, 'test_data', 'attributes_data.csv')
    attr = pd.read_csv(attr_file_path, dtype={'SEC_ID': 'str', 'IND_CD': 'str'})
    attr.set_index('SEC_ID', inplace=True, drop=True)

    dao = Dao.init_from_dataframes(covar=covar, attributes=attr)

    weights = {sec_id: 1.0 / len(dao.get_securities_list()) for sec_id in dao.get_securities_dict()}

    port = Portfolio(weights, dao.get_securities_list(), dao.get_covar_matrix())

    # produce_params_qp(...)
    qp_params = Objective.produce_params_qp(ptf=port, risk_tol=1)
    assert np.allclose(qp_params.get('Q'), port.get_covar_matrix().to_numpy())
    assert np.allclose(list(qp_params.get('p')), -port.to_dataframe()['RETURN'].to_numpy(), atol=0.01)

    risk_params = Objective.produce_params_risk(ptf=port)
    assert np.allclose(port.get_covar_matrix().to_numpy().astype(np.double),
                       risk_params.get('COVAR'))

    rtrn_params = Objective.produce_params_rtrn(ptf=port)
    assert np.allclose([port.get_security(sec_id).get_attr_value('return') for sec_id in port.get_id_list()],
                       rtrn_params.get('RETURN'))

    sharpe_params = Objective.produce_params_sharpe(ptf=port, bmk=0.01)
    assert np.allclose(port.get_covar_matrix().to_numpy().astype(np.double), sharpe_params.get('Q'))
    assert np.allclose([port.get_security(sec_id).get_attr_value('return') for sec_id in port.get_id_list()],
                       sharpe_params.get('p'))


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        test()
    except Exception as err:
        print("Objective unit port failed: " + str(err))
        print(traceback.format_exc())