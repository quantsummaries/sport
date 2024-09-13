import traceback

import pandas as pd

import ut_constraint
import ut_covar_estimator
import ut_dao
import ut_functions
import ut_objective
import ut_optimizer
import ut_portfrolio
import ut_security


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        print("---port.ut_covar_estimator.port()")
        ut_covar_estimator.test()

        print("---port.ut_security.port()")
        ut_security.test()

        print("---port.ut_portfolio.port()")
        ut_portfrolio.test()

        print("---port.ut_dao.port()")
        ut_dao.test()

        print("---port.ut_functions.port()")
        ut_functions.test()

        print("---port.ut_constraint.port()")
        ut_constraint.test()

        print("---port.ut_objective.port()")
        ut_objective.test()

        print("---port.ut_optimizer.port()")
        ut_optimizer.test()

    except Exception as err:
        print('Testing all unit tests failed: ' + str(err))
        print(traceback.format_exc())
