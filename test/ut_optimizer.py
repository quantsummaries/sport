import os
import traceback

from cvxopt import matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint

from sport import Dao, Constraint, Objective, Optimizer, Portfolio
from sport.functions import obj_neg_rtrn, obj_neg_sharpe_ratio, obj_qp, obj_risk


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

    weights = {sec_id: 1.0 / len(dao.get_securities_list()) for sec_id in dao.get_securities_dict()}
    port = Portfolio(weights, dao.get_securities_list(), dao.get_covar_matrix())
    x0 = [1.0 / len(port.get_id_list())] * len(port.get_id_list())

    cfg_file_path_csv = os.path.join(dir_path, 'test_data', 'constraints_cfg.csv')
    constr = Constraint({'Portfolio': port, 'Data': cfg_file_path_csv, 'DataFormat': 'csv'})

    test_qp_basic()
    test_qp_port(port, x0, constr)
    test_risk_obj(port, x0, constr)
    test_rtrn_obj(port, x0, constr)
    test_sharpe_port(port, x0, constr)
    test_mean_variance(port, x0)

    cfg_file_path_csv_risk = os.path.join(dir_path, 'test_data', 'constraints_cfg_nonlinear_risk.csv')
    constr_risk = Constraint.init_from_table(ptf=port, data=cfg_file_path_csv_risk, data_format='csv')
    test_risk_constr(port, x0, constr_risk)


def test_qp_basic():
    """Unit port of basic example from cvxopt https://cvxopt.org/examples/tutorial/qp.html."""
    print("--------- begin of test_qp_basic() ---------")

    # _opt_cvxopt_solvers_qp()
    Q = matrix([[2.0, 0.5], [0.5, 1.0]])
    p = matrix([1.0, 1.0])

    G = matrix([[-1.0, 0.0], [0.0, -1.0]])
    h = matrix([0.0, 0.0])
    A = matrix([1.0, 1.0], (1, 2))
    b = matrix([1.0])

    constr_cvxopt = Constraint({'G': G, 'h': h, 'A': A, 'b': b})
    obj_cvxopt = Objective({'Q': Q, 'p': p})
    opt_cvxopt = Optimizer(x0=[0.0, 0.0], obj=obj_cvxopt, constr=constr_cvxopt)

    res_cvxopt = opt_cvxopt.optimize(method='CVXOPT_SOLVERS_QP')

    assert np.allclose([0.25, 0.75], res_cvxopt, atol=1e-05)

    # _opt_scipy_optimize_minimize_trust_constr()
    bounds = Bounds([0, 0], [np.inf, np.inf])
    linear_constr = LinearConstraint([[1,1]], [1], [1])

    constr_scipy = Constraint({'Bounds': bounds, 'LinearConstraint': linear_constr, 'NonlinearConstraint': None})

    obj_scipy = Objective({'Function': obj_qp, 'Params': {'Q': Q, 'p': p}})

    opt_scipy = Optimizer(x0=[0.0, 0.0], obj=obj_scipy, constr=constr_scipy)

    res_scipy = opt_scipy.optimize(method='SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR')

    assert np.allclose([0.25, 0.75], res_scipy, atol=1e-08)

    print("--------- end of test_qp_basic() ---------")
    print()


def test_qp_port(port, x0, constr):
    """Use real portfolio to port the QP optimization."""

    print("--------- begin of test_qp_port() ---------")

    obj = Objective({'Function': obj_qp, 'Params': Objective.produce_params_qp(ptf=port, risk_tol=1)})
    opt = Optimizer(x0=x0, obj=obj, constr=constr)

    print(constr.to_dataframe())
    print()

    res_cvxopt = opt.optimize(method='CVXOPT_SOLVERS_QP')
    port_cvxopt = port.reproduce_by_wts(res_cvxopt)
    output_cvxopt = port_cvxopt.get_weights_dataframe()
    output_cvxopt['NAME'] = [sec.get_attr_value('SEC_NM') for sec in port.get_securities()]
    output_cvxopt['RETURN'] = [round(sec.get_attr_value('RETURN'), 3) for sec in port.get_securities()]
    output_cvxopt['RISK'] = [round(sec.get_attr_value('RISK'), 3) for sec in port.get_securities()]
    output_cvxopt['DIV_YLD'] = [round(sec.get_attr_value('DIV_YLD'), 3) for sec in port.get_securities()]

    print('*** mean-variance optimal portfolio:')
    print(output_cvxopt)
    print()

    res_scipy = opt.optimize(method='SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR')
    port_scipy = port.reproduce_by_wts(res_scipy)
    assert np.allclose(port_cvxopt.get_weights_list(), port_scipy.get_weights_list(), atol=1e-03)

    print("--------- end of test_qp_port() ---------")
    print()


def test_risk_obj(port, x0, constr):
    """Unit port of risk objective function."""
    print("--------- begin of test_risk_obj() ---------")

    obj = Objective({'Function': obj_risk, 'Params': Objective.produce_params_risk(port)})

    opt = Optimizer(x0=x0, obj=obj, constr=constr)

    print(constr.to_dataframe())
    print()

    res = opt.optimize(method='SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR')

    port = port.reproduce_by_wts(res)
    output_scipy = port.get_weights_dataframe()
    output_scipy['NAME'] = [sec.get_attr_value('SEC_NM') for sec in port.get_securities()]
    output_scipy['RETURN'] = [round(sec.get_attr_value('RETURN'), 3) for sec in port.get_securities()]
    output_scipy['RISK'] = [round(sec.get_attr_value('RISK'), 3) for sec in port.get_securities()]

    print('*** Returns optimal portfolio:')
    print(output_scipy)
    print("Portfolio's risk: " + str(port.get_risk()))

    print("--------- end of test_risk_obj() ---------")
    print()


def test_rtrn_obj(port, x0, constr):
    """Unit port of negative of returns objective function."""
    print("--------- begin of test_rtrn_obj() ---------")

    obj = Objective({'Function': obj_neg_rtrn, 'Params': Objective.produce_params_rtrn(port)})

    opt = Optimizer(x0=x0, obj=obj, constr=constr)

    print(constr.to_dataframe())
    print()

    res = opt.optimize(method='SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR')

    port = port.reproduce_by_wts(res)
    output_scipy = port.get_weights_dataframe()
    output_scipy['NAME'] = [sec.get_attr_value('SEC_NM') for sec in port.get_securities()]
    output_scipy['RETURN'] = [round(sec.get_attr_value('RETURN'), 3) for sec in port.get_securities()]
    output_scipy['RISK'] = [round(sec.get_attr_value('RISK'), 3) for sec in port.get_securities()]
    output_scipy['DIV_YLD'] = [round(sec.get_attr_value('DIV_YLD'), 3) for sec in port.get_securities()]

    print('*** Returns optimal portfolio:')
    print(output_scipy)
    print("Portfolio's return: " + str(port.get_attr_value('return')))

    print("--------- end of test_rtrn_obj() ---------")
    print()


def test_sharpe_port(port, x0, constr):
    """Unit port of Sharpe ratio objective function."""
    print("--------- begin of test_sharpe_port() ---------")

    benchmark = 0.01
    obj = Objective({'Function': obj_neg_sharpe_ratio, 'Params': Objective.produce_params_sharpe(ptf=port,
                                                                                                 bmk=benchmark)})

    opt = Optimizer(x0=x0, obj=obj, constr=constr)

    print(constr.to_dataframe())
    print()

    res = opt.optimize(method='SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR')

    port = port.reproduce_by_wts(res)
    output_scipy = port.get_weights_dataframe()
    output_scipy['NAME'] = [sec.get_attr_value('SEC_NM') for sec in port.get_securities()]
    output_scipy['RETURN'] = [round(sec.get_attr_value('RETURN'), 3) for sec in port.get_securities()]
    output_scipy['RISK'] = [round(sec.get_attr_value('RISK'), 3) for sec in port.get_securities()]
    output_scipy['DIV_YLD'] = [round(sec.get_attr_value('DIV_YLD'), 3) for sec in port.get_securities()]
    output_scipy['SHARPE_RATIO'] = [(x - benchmark) / y if y > 0 else None for x, y in zip(output_scipy['RETURN'], output_scipy['RISK'])]

    print('*** Sharpe ratio optimal portfolio:')
    print(output_scipy)
    print("Portfolio's Sharpe ratio: " + str(port.get_sharpe_ratio(bmk=benchmark)))
    print()

    # compare with previous allocation to see optimal sharpe ratio
    wts = port.get_weights_dict()
    wts['600276'] = wts['300347'] + wts['600276']
    wts['300347'] = 0
    new_wts = [wts.get(id) for id in sorted(list(wts.keys()))]
    new_port = port.reproduce_by_wts(new_wts)
    new_output = new_port.get_weights_dataframe()
    new_output['NAME'] = [sec.get_attr_value('SEC_NM') for sec in port.get_securities()]
    print("*** What if 300347's weight is allocated to 600276? Sharpe ratio is nonlinear: " + str(new_port.get_sharpe_ratio(bmk=benchmark)))
    print(new_output)

    print("--------- end of test_sharpe_port() ---------")
    print()


def test_mean_variance(port, x0):
    print("--------- begin of test_mean_variance() ---------")

    constr_data = pd.DataFrame.from_dict({'TYPE': ['LINEAR_INEQ', 'LINEAR_EQ'],
                                          'SEC_ID': ['EACH', 'ALL'],
                                          'ATTRIBUTE': ['WEIGHT', 'WEIGHT'],
                                          'ATTRIBUTE_PARAMS': [None, None],
                                          'VALUE':['[0.0:1.0]', '1']})
    constr_mv = Constraint({'Portfolio': port, 'Data': constr_data, 'DataFormat': 'dataframe'})
    print(constr_mv.to_dataframe())

    print("\n*** Plot mean-variance efficient frontier:")
    risk = list()
    rtrn = list()
    vals = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    for risk_tol in vals:
        obj = Objective({'Function': obj_qp, 'Params': Objective.produce_params_qp(ptf=port, risk_tol=risk_tol)})
        opt = Optimizer(x0=x0, obj=obj, constr=constr_mv)
        res = opt.optimize(method='CVXOPT_SOLVERS_QP')

        optimal_ptf = port.reproduce_by_wts(res)

        #output_cvxopt = optimal_ptf.get_weights_dataframe()
        #output_cvxopt['NAME'] = [sec.get_attr_value('SEC_NM') for sec in port.get_securities()]
        #output_cvxopt['RETURN'] = [round(sec.get_attr_value('RETURN'), 3) for sec in port.get_securities()]
        #output_cvxopt['RISK'] = [round(sec.get_attr_value('RISK'), 3) for sec in port.get_securities()]
        #output_cvxopt['DIVIDEND_YLD'] = [round(sec.get_attr_value('DIVIDEND_YLD'), 3) for sec in port.get_securities()]
        #print(output_cvxopt)

        risk.append(optimal_ptf.get_risk())
        rtrn.append(optimal_ptf.get_attr_value('RETURN'))

        print('risk_tol=' + str(risk_tol) + ', return=' + str(round(optimal_ptf.get_attr_value('RETURN'), 2)) +
              ', risk=' + str(round(optimal_ptf.get_risk(), 2)))

    plt.plot(risk, rtrn, 'ro', risk, rtrn, 'k')
    plt.xlabel('RISK')
    plt.ylabel('RETURN')
    plt.title('Efficient Frontier of MV Optimization')

    dir_path = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(dir_path, 'test_output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    plt.savefig(os.path.join(output_dir, 'efficient_frontier.png'))
    plt.close()

    print("--------- end of test_mean_variance() ---------")
    print()


def test_risk_constr(port, x0, constr_risk):
    """Unit port of nonlinear constraint: risk."""
    print("--------- begin of test_risk_constr() ---------")

    obj = Objective({'Function': obj_qp, 'Params': Objective.produce_params_qp(ptf=port, risk_tol=10.0)})

    opt = Optimizer(x0=x0, obj=obj, constr=constr_risk)

    print(constr_risk.to_dataframe())
    print()

    res = opt.optimize(method='SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR')

    port = port.reproduce_by_wts(res)
    output_scipy = port.get_weights_dataframe()
    output_scipy['NAME'] = [sec.get_attr_value('SEC_NM') for sec in port.get_securities()]
    output_scipy['RETURN'] = [round(sec.get_attr_value('RETURN'), 3) for sec in port.get_securities()]
    output_scipy['RISK'] = [round(sec.get_attr_value('RISK'), 3) for sec in port.get_securities()]

    print(output_scipy)
    print("Portfolio's risk: " + str(port.get_risk()))
    print("300347's risk: " + str(port.get_risk(sec_ids=['300347'])))
    print("--------- end of test_risk_constr() ---------")
    print()


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        test()
    except Exception as err:
        print('Optimizer unit port failed: ' + str(err))
        print(traceback.format_exc())