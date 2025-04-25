# Python standard library: https://docs.python.org/3/library/index.html
import datetime
import os
import traceback

# open source packages
import pandas as pd

# local packages
from sport import CovarEstimator


def enrich_attributes(covar_estimator: CovarEstimator, attributes_file: str):
    attributes = pd.read_csv(attributes_file).set_index('SEC_ID', drop=True)
    for sec_id, attributes_dict in covar_estimator._sec_id_to_last_data.items():
        if sec_id not in attributes.index:
            print(f"""WARNING: SEC_ID '{sec_id}' does not have attributes data in {attributes_file}""")
            continue
        attributes_dict['SEC_NM'] = attributes.loc[sec_id, 'SEC_NM']


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        dir_path = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(dir_path, 'data', 'etf')

        covar_estimator = CovarEstimator(data_dir=data_dir,
                                         rtrn_method='logarithm',
                                         halflife_in_yrs=1.0,
                                         cash_rtrn=0.0,
                                         price_type='CLOSE',
                                         sec_id_list=['SPY', 'VTV', 'VBR', 'VTI', 'VWO', 'BND', 'AGG', 'JNK', 'TIP', 'VNQ', 'GSG', 'DBC', 'GLD'],
                                         start_dt=datetime.date(2007, 12, 11),
                                         end_dt=datetime.date(2025, 2, 11))

        enrich_attributes(covar_estimator=covar_estimator,
                          attributes_file=os.path.join(dir_path, 'data', 'etf_attributes_data.csv'))

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