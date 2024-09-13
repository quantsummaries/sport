# Script to generate additional data for consumption

import os
import traceback

import pandas as pd


if __name__ == '__main__':
    try:
        # industry classification
        industry_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'additional', 'industry_shenwan.csv')
        industry = pd.read_csv(industry_path, dtype={'行业代码': 'str', '行业名称': 'str', '股票代码': 'str', '股票名称': 'str'})
        industry.columns = ['IND_CD', 'IND_NM', 'SEC_ID', 'SEC_NM']
        industry.set_index(keys=['SEC_ID'], drop=True, inplace=True)

        # dividend yield: fake data
        dividend = pd.DataFrame({'SEC_ID': ['000001', '002138', '002777', '300347',
                                            '300502', '600030', '600276', '000000'],
                                 'DIV_YLD': [0.05376582276605168, 0.05809967689598314,
                                             0.0903734138106532, 0.09972711780656822,
                                             0.025927879622836617, 0.07283933892055781,
                                             0.05577039848123399, 0.0]
                                 })
        dividend.set_index(keys=['SEC_ID'], drop=True, inplace=True)

        # final data
        attributes = industry.merge(dividend, how='outer', on='SEC_ID')
        output_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'attributes_data.csv')
        attributes.to_csv(output_file)

        print(f"""data saved to {output_file}""")
    except Exception as err:
        print('Attributes data batch failed: ' + str(err))
        print(traceback.format_exc())