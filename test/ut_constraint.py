import os
import traceback

import numpy as np
import pandas as pd

from sport import Dao, Constraint, Portfolio


def test():
    """Unit tests."""
    dir_path = os.path.abspath(os.path.dirname(__file__))

    covar_file_path = os.path.join(dir_path, 'test_data', 'covar_matrix.csv')
    covar = pd.read_csv(covar_file_path, dtype={'SEC_ID': 'str'}).set_index(keys=['SEC_ID'], drop=True)

    attr_file_path = os.path.join(dir_path, 'test_data', 'attributes_data.csv')
    attr = pd.read_csv(attr_file_path, dtype={'SEC_ID': 'str', 'IND_CD': 'str'}).set_index(keys=['SEC_ID'], drop=True)

    dao = Dao.init_from_dataframes(covar, attr)

    weights = {sec_id: 1.0 / len(dao.get_securities_list()) for sec_id in dao.get_securities_dict()}

    port = Portfolio(weights, dao.get_securities_list(), dao.get_covar_matrix())

    # __init__(...)
    cfg_file_path_csv = os.path.join(dir_path, 'test_data', 'constraints_cfg.csv')
    cfg_file_path_json = os.path.join(dir_path, 'test_data', 'constraints_cfg.json')

    constr1 = Constraint.init_from_table(ptf=port, data=cfg_file_path_csv, data_format='csv')
    constr2 = Constraint.init_from_table(ptf=port, data=cfg_file_path_json, data_format='json')
    constr3 = Constraint.init_from_table(ptf=port, data=pd.read_csv(cfg_file_path_csv), data_format='dataframe')

    # constr1 == constr2
    assert set(constr1.to_dataframe().index) == set(constr2.to_dataframe().index)
    assert set(constr1.to_dataframe().columns) == set(constr2.to_dataframe().columns)
    for col_nm in constr1.to_dataframe().columns:
        for idx in constr1.to_dataframe().index:
            if str(constr1.to_dataframe().loc[idx, col_nm]) == 'nan' and str(constr2.to_dataframe().loc[idx, col_nm]) == '':
                continue
            assert str(constr1.to_dataframe().loc[idx, col_nm]) == str(constr2.to_dataframe().loc[idx, col_nm])

    # constr1 == constr3
    assert constr1.to_dataframe().equals(constr3.to_dataframe())

    # Constraint._parse_bounds(...)
    lb, ub = Constraint._parse_bounds('[1.0:2.5]')
    assert np.isclose(lb, 1.0)
    assert np.isclose(ub, 2.5)

    # _get_idx_from_id(...)
    assert constr1._get_idx_from_id('600030') == 5

    # to_dataframe()
    print('\n------- linear constraints in a table')
    print(constr1.to_dataframe())
    #print(constr1.to_dataframe(raw=False))

    # get_constraints_cvxopt_solvers_qp()
    constr_cvxopt = constr1.get_constr_cvxopt_solvers_qp()
    print('\n---G:')
    print(constr_cvxopt.get('G'))
    print('\n---h')
    print(constr_cvxopt.get('h'))
    print('\n---A')
    print(constr_cvxopt.get('A'))
    print('\n---b')
    print(constr_cvxopt.get('b'))

    # get_constraints_scipy_optimize_minimize_trust_constr()
    constr_scipy = constr1.get_constr_scipy_optimize_minimize_trust_constr()
    print('\n---Bounds')
    print(constr_scipy.get('Bounds'))
    print('\n---LinearConstraint.A')
    print(constr_scipy.get('LinearConstraint').A)
    print('\n---LinearConstraint.lb')
    print(constr_scipy.get('LinearConstraint').lb)
    print('\n---LinearConstraint.ub')
    print(constr_scipy.get('LinearConstraint').ub)

    # port nonlinear constraints
    cfg_file_path_csv_nonlinear = os.path.join(dir_path, 'test_data', 'constraints_cfg_nonlinear_risk.csv')
    constr4 = Constraint.init_from_table(ptf=port,
                                         data=cfg_file_path_csv_nonlinear,
                                         data_format='csv')

    # to_dataframe()
    print('\n------- linear & nonlinear constraints in a table')
    print(constr4.to_dataframe())
    #print(constr4.to_dataframe(raw=False))

    constr_nonlinear = constr4.get_constr_scipy_optimize_minimize_trust_constr()
    print('\n---Bounds')
    print(constr_nonlinear.get('Bounds'))
    print('\n---LinearConstraint.A')
    print(constr_nonlinear.get('LinearConstraint').A)
    print('\n---LinearConstraint.lb')
    print(constr_nonlinear.get('LinearConstraint').lb)
    print('\n---LinearConstraint.ub')
    print(constr_nonlinear.get('LinearConstraint').ub)
    print('\n---NonlinearConstraint.fun')
    print(constr_nonlinear.get('NonlinearConstraint').fun)
    print('\n---NonlinearConstraint.lb')
    print(constr_nonlinear.get('NonlinearConstraint').lb)
    print('\n---NonlinearConstraint.ub')
    print(constr_nonlinear.get('NonlinearConstraint').ub)
    print('\n---NonlinearConstraint.jac')
    print(constr_nonlinear.get('NonlinearConstraint').jac)
    print('\n---NonlinearConstraint.hess')
    print(constr_nonlinear.get('NonlinearConstraint').hess)

    # verify nonlinear constraint function is successfully constructed
    covar = port.get_covar_matrix().to_numpy()
    mask1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    mask2 = np.array([0, 0, 0, 1, 0, 0, 0, 0]) # sec_id='300347'
    x = [0.2, 0.1, 0.3, 0.25, 0.05, 0.04, 0.05, 0.01]
    v1 = np.array(x) * mask1
    v2 = np.array(x) * mask2
    y1 = np.sqrt(np.matmul(np.matmul(v1, covar), v1.transpose()))
    y2 = np.sqrt(np.matmul(np.matmul(v2, covar), v2.transpose()))
    f = constr_nonlinear.get('NonlinearConstraint').fun
    y = f(x)
    assert np.allclose([y1, y2], y)

    # port classification constraint
    cfg_file_path_csv_ind = os.path.join(dir_path, 'test_data', 'constraints_cfg_ind.csv')
    constr5 = Constraint.init_from_table(ptf=port,
                                         data=cfg_file_path_csv_ind,
                                         data_format='csv')
    print('\n------- Classification constraint')
    print(constr5.to_dataframe(raw=True))
    print(constr5.to_dataframe(raw=False))

    constr_ind = constr5.get_constr_scipy_optimize_minimize_trust_constr()
    print('\n---Bounds')
    print(constr_ind.get('Bounds'))
    print('\n---LinearConstraint.A')
    print(constr_ind.get('LinearConstraint').A)
    print('\n---LinearConstraint.lb')
    print(constr_ind.get('LinearConstraint').lb)
    print('\n---LinearConstraint.ub')
    print(constr_ind.get('LinearConstraint').ub)

if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)
        test()
    except Exception as err:
        print('Constraint unit test failed: ' + str(err))
        print(traceback.format_exc())