import os
import traceback

import pandas as pd

from sport import Constants, Constraint, Dao, Objective, Optimizer, Portfolio


def calc_optimal_wts(dao, sec_id_list_input, returns_input, covar_input, constr_input, objective, param):
    """Calculate optimal weights.

    Args:
        dao (Dao): a Data Access Object.
        sec_id_list_input (list): a list of security IDs.
        returns_input (dict): {security ID: return}.
        covar_input (pandas.DataFrame): covariance matrix collected from input file.
        constr_input (pandas.DataFrame): constraints table collected from input file.
        objective (str): 'MEAN_VARIANCE', 'MAX_SHARPE_RATIO', etc.
        param (double): parameter for objective function, e.g. risk tolerance, benchmark rate, etc.
    """
    # exclude cash from security list if objective is 'MAX_SHARPE_RATIO' or 'RISK_PARITY'
    returns = returns_input.copy()
    covar = covar_input.copy()
    if objective in {'MAX_SHARPE_RATIO', 'RISK_PARITY'} and '000000' in sec_id_list_input:
        sec_id_list = [x for x in sec_id_list_input if x != '000000']

        if '000000' in returns:
            del returns['000000']

        if '000000' in covar.index:
            covar.drop(labels=['000000'], axis='index', inplace=True)
            covar.drop(labels=['000000'], axis='columns', inplace=True)
    else:
        sec_id_list = sec_id_list_input.copy()

    init_wts_dict = {x: 1/len(sec_id_list) for x in sec_id_list}

    port = Portfolio(init_wts_dict, dao.get_securities_list(sec_id_list), dao.get_covar_matrix(sec_id_list))
    port = port.reproduce_by_attr('RETURN', returns)
    port = port.reproduce_by_covar(covar)

    constr = Constraint.init_from_table(ptf=port,
                                        data=constr_input,
                                        data_format='dataframe')

    if objective == 'MEAN_VARIANCE':
        obj_fun = Objective.init_mean_varinace_obj(ptf=port, risk_tol=param)
    elif objective == 'MAX_SHARPE_RATIO':
        obj_fun = Objective.init_sharpe_ratio_obj(ptf=port, benchmark=param)
    elif objective == 'MAX_RETURN':
        obj_fun = Objective.init_rtrn_obj(ptf=port)
    elif objective == 'MIN_RISK':
        obj_fun = Objective.init_risk_obj(ptf=port)
    elif objective == 'RISK_PARITY':
        obj_fun = Objective.init_risk_parity_obj(ptf=port)
    elif objective == 'MIN_AVG_MAX_DRAWDOWN':
        obj_fun = Objective.init_avg_max_drawdown_obj(ptf=port, num_days=param)
    else:
        raise Exception(objective + ' is not a supported objective function.')

    optimizer = Optimizer(x0=port.get_weights_list(), obj=obj_fun, constr=constr)
    opt_wt_list = optimizer.optimize(method='SCIPY_OPTIMIZE_MINIMIZE_TRUST_CONSTR')

    port_optimal = port.reproduce_by_wts(opt_wt_list)
    opt_result, opt_analytics = port_optimal.report(obj_type=objective, obj_param=param, addtl=['IND_NM'])

    return opt_result, opt_analytics


def process_input_output(dao, uuid):
    """Collect input, send it to calculator, and process output.

    Args:
        dao (Dao): a data access object.
        uuid (str): UUID used to name input and output files.
    """
    input_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), uuid + '_input.xlsx')
    output_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), uuid + '_output.xlsx')

    sec_id_list_input, returns_input, covar_input, objectives_input, constr_df = Dao.read_excel_input(input_file_path)

    print("\n----- INPUT -----\n")
    print(constr_df)

    results = dict()

    print("\n----- OUTPUT -----\n")
    summary = list()
    for objective in objectives_input:
        result, analytics = calc_optimal_wts(dao,
                                             sec_id_list_input,
                                             returns_input,
                                             covar_input,
                                             constr_df,
                                             objective,
                                             param=objectives_input.get(objective))
        result.reset_index(inplace=True)
        result.columns = [Constants.en_to_cn.get(x) for x in result.columns]
        results[Constants.en_to_cn.get(objective)] = result

        summary.append(analytics)

        print(f'\n{objective}')
        print(result)

    summary = pd.concat(summary)
    summary.reset_index(inplace=True, drop=True)
    summary['OBJECTIVE'] = [Constants.en_to_cn.get(x) for x in summary['OBJECTIVE']]
    summary['ANALYTICS'] = [Constants.en_to_cn.get(x) for x in summary['ANALYTICS']]
    summary.columns = [Constants.en_to_cn.get(x) for x in summary.columns]

    results['总结'] = summary
    print('\nSummary')
    print(summary)

    Dao.write_to_excel(output_file_path, results)

    print(f"""loading input from {input_file_path}""")
    print(f"""saving output to {output_file_path}""")


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        covar_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'covar_matrix.csv')
        attributes_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'attributes_data.csv')
        dao = Dao.init_from_default_data(covar_path=covar_path, attributes_path=attributes_path)

        # read inputs
        all_uuid = ["template"]
        for uuid in all_uuid:
            process_input_output(dao, uuid)

    except Exception as err:
        print('Main batch failed: ' + str(err))
        print('尊敬的客户，现在无优化结果。请检查输入的表头是否与模板一直。输入数据中是否存在异常报错的值，是否输入绝对值或者忘带百分比%符号，或异常的限制参数等。请调整后重新输入。')
        print(traceback.format_exc())
