# standard packages
import math
import os
import traceback
from typing import List

# open source packages
import pandas as pd


def drawdown(rtrns: List[float], type: str) -> pd.DataFrame:
    """Calculate the drawdown of a return series.

    Args:
        rtrns (List[float]): a time series of returns.
        type (str): return type, 'simple' or 'log'.
    Returns:
        drawdown (pd.DataFrame): a dataframe of 'cum_rtrn', 'max_cum_rtrn', and 'drawdown'.
    """
    if rtrns is None or not isinstance(rtrns, list) or len(rtrns) == 0:
        raise ValueError("input returns are not valid")

    if type is None or not isinstance(type, str) or type not in ('simple', 'log'):
        raise ValueError("input return type is not valid")

    cum_rtrns = [0]
    curr_cum_rtrn = cum_rtrns[-1]
    max_cum_rtrns = [0]
    curr_max_cum_rtrn = max_cum_rtrns[-1]

    for i in range(len(rtrns)):
        if type == 'log':
            curr_cum_rtrn = curr_cum_rtrn + rtrns[i]
        elif type == 'simple':
            curr_cum_rtrn = (curr_cum_rtrn + 1) * (rtrns[i] + 1) - 1

        cum_rtrns.append(curr_cum_rtrn)

        if curr_max_cum_rtrn < curr_cum_rtrn:
            curr_max_cum_rtrn = curr_cum_rtrn

        max_cum_rtrns.append(curr_max_cum_rtrn)

    drawdown = pd.DataFrame({'cum_rtrn': cum_rtrns, 'max_cum_rtrn': max_cum_rtrns})
    drawdown['drawdown'] = drawdown['max_cum_rtrn'] - drawdown['cum_rtrn']

    return drawdown


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        ticker = 'SPY'
        file_spy = os.path.join(os.getcwd(), '..', 'test', 'test_data', 'etf', f"""{ticker}.csv""")
        data = pd.read_csv(file_spy)

        prices = data['CLOSE'].to_list()
        rtrns = [math.log(prices[i]) - math.log(prices[i - 1]) for i in range(1, len(prices))]

        dd = drawdown(rtrns=rtrns, type='log')
        dd.index = data['DATE']
        print(dd.head())

    except Exception as err:
        print("Objective unit port failed: " + str(err))
        print(traceback.format_exc())
