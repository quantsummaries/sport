import os
import traceback

import pandas as pd

from sport import CovarEstimator


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        dir_path = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(dir_path, 'data', 'stockfiles')

        covar_estimator = CovarEstimator(data_dir=data_dir, rtrn_method='logarithm', halflife_in_yrs=1.0)

        output = covar_estimator.to_dataframe()
        output_file = os.path.join(dir_path, 'data', 'covar_matrix.csv')
        output.to_csv(output_file, index=True)

        print(output)
        print(f"""data saved to {output_file}""")

    except Exception as err:
        print('Covariance estimation batch failed: ' + str(err))
        print(traceback.format_exc())