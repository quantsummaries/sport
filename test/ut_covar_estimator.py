# -*- coding: utf-8 -*-

import os
import traceback

import pandas as pd

from sport import CovarEstimator


def test():
    """Unit port"""
    dir_path = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(dir_path, 'test_data', 'stockfiles')

    covar_estimator = CovarEstimator(data_dir=data_dir, rtrn_method='logarithm', halflife_in_yrs=1.0)

    output = covar_estimator.to_dataframe()

    output.to_csv(os.path.join(dir_path, 'test_output', 'covar_matrix.csv'), index=True)

    print(output)


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)
        test()
    except Exception as err:
        print("CovarEstimator unit port failed: ", err)
        print(traceback.format_exc())
