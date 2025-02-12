import os
import traceback

import pandas as pd

from sport import CovarEstimator


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        dir_path = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(dir_path, 'data', 'etf')

        covar_estimator = CovarEstimator(data_dir=data_dir, rtrn_method='logarithm', halflife_in_yrs=1.0, cash_rtrn=0.0)

        covar_file = os.path.join(dir_path, 'data', 'etf_covar_matrix.csv')
        covar_estimator.to_dataframe().to_csv(covar_file, index=True)

        corr_file = os.path.join(dir_path, 'data', 'etf_corr_matrix.csv')
        covar_estimator.get_corr_matrix().to_csv(corr_file)

        print(covar_estimator.to_dataframe())
        print(covar_estimator.get_corr_matrix())
        print(f"""data saved to {covar_file} and {corr_file}""")

    except Exception as err:
        print('Covariance estimation batch failed: ' + str(err))
        print(traceback.format_exc())