# standard packages
import datetime
import math
import os
import traceback
from typing import List

# open source packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def drawdown(rtrns: List[float], type: str) -> pd.DataFrame:
    """Calculate the drawdown of a return series.

    Args:
        rtrns (List[float]): a time series of returns.
        type (str): return type, 'simple' or 'log'.
    Returns:
        drawdown (pd.DataFrame): a dataframe of three columns, 'cum_rtrn', 'max_cum_rtrn', and 'drawdown'.
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


def calc_rtrns_all(tickers: List[str],
                   data_dir: str,
                   start_dt: datetime.date,
                   end_dt: datetime.date,
                   type: str='logarithm') -> pd.DataFrame:
    """Calculate returns for given tickers.
    Args:
        tickers (List[str]): list of tickers.
        data_dir (str): data directory.
        start_dt (datetime.date): start date of the return calculation.
        end_dt (datetime.date): end date of the return calculation.
        type (str): return calculation method, 'simple' or 'logarithm'.
    Returns:
        rtrns (pandas.DataFrame): data frame of returns indexed by 'DATE', one column per ticker.
    """
    if tickers is None or not isinstance(tickers, list) or len(tickers) == 0:
        raise ValueError(f"""Input tickers is not valid: {tickers}""")
    if data_dir is None or not isinstance(data_dir, str) or not os.path.exists(data_dir):
        raise ValueError(f"""Input data_dir is not valid: {data_dir}""")
    if start_dt is None or not isinstance(start_dt, datetime.date):
        raise ValueError(f"""Input start_dt is not valid: {start_dt}""")
    if end_dt is None or not isinstance(end_dt, datetime.date):
        raise ValueError(f"""Input start_dt is not valid: {end_dt}""")
    if start_dt >= end_dt:
        raise ValueError(f"""start_dt {start_dt} is not before end_dt {end_dt}""")
    if type is None or not isinstance(type, str) or type not in ('simple', 'logarithm'):
        raise ValueError(f"""Input type is not valid: {type}""")

    prices = dict()
    for ticker in tickers:
        df = pd.read_csv(os.path.join(data_dir, f"""{ticker}.csv"""),
                         dtype={'DATE': 'str', 'CLOSE': 'float', 'HIGH': 'float', 'LOW': 'float', 'OPEN': 'float',
                                'VOLUME': 'int','SEC_ID': 'str'})
        df['DATE'] = [datetime.datetime.strptime(x, "%Y-%m-%d").date() for x in df['DATE']]

        if df['DATE'].iloc[0] > start_dt or df['DATE'].iloc[-1] < end_dt:
            raise ValueError(f"""{ticker} does not have sufficient data to cover [{start_dt}, {end_dt}]: [{df['DATE'].iloc[0]}, {df['DATE'].iloc[-1]}]""")

        df = df[(df['DATE'] >= start_dt) & (df['DATE'] <= end_dt)]

        if df['CLOSE'].isna().any():
            raise ValueError(f"""{ticker} has NA in 'CLOSE': \n{df[df['CLOSE'].isna()]}""")

        if 'DATE' not in prices:
            prices['DATE'] = df['DATE'].to_list()

        prices[ticker] = df['CLOSE'].to_list()

    prices = pd.DataFrame(prices)

    rtrns = pd.DataFrame({'DATE': prices['DATE'].to_list()[1:]})
    for ticker in prices.columns:
        if ticker == 'DATE':
            continue
        px = prices[ticker].to_list()
        if type == 'simple':
            rtrns[ticker] = [px[i] / px[i - 1] - 1.0 for i in range(1, len(px))]
        elif type == 'logarithm':
            rtrns[ticker] = [math.log(px[i]) - math.log(px[i - 1]) for i in range(1, len(px))]

    rtrns.set_index('DATE', drop=True, inplace=True)

    return rtrns


def calc_port_rtrns(rtrns, weights) -> pd.DataFrame:
    """Calculate portfolio return for given weights.
    Args:
        rtrns (pandas.DataFrame): a dataframe of returns, each column is the returns of a ticker.
        weights (Dict[str, float]): a dictionary of weights, {ticker: weight, ...}.
    Returns:
        port_rtrns (pandas.DataFrame): a dataframe of returns, columns include portfolio returns ('Portfolio') and individual returns.
    """
    if rtrns is None or not isinstance(rtrns, pd.DataFrame) or rtrns.shape[0] == 0:
        raise ValueError("Input rtrns is not valid")
    if weights is None or not isinstance(weights, dict) or len(weights) == 0:
        raise ValueError("Input weights is not valid")

    for k in weights.keys():
        if k not in rtrns.columns:
            raise ValueError(f"""{k} is not in returns dataframe:\n {rtrns.columns}""")

    port_rtrns = rtrns[list(weights.keys())].copy()
    port_rtrns['Portfolio'] = 0
    for ticker in weights.keys():
        port_rtrns['Portfolio'] = port_rtrns['Portfolio'] + weights[ticker] * port_rtrns[ticker]

    return port_rtrns


def find_rolling_rtrns(data_dir: str, ticker: str, start_dt: datetime.date, step_tenor: int, span_tenor: int) -> pd.DataFrame:
    """ Calculate returns in rolling windows. E.g. [t1, t1+span], [t2, t2+span], etc.
    Args:
        data_dir (str): data directory.
        ticker (str): security ID.
        start_dt (datetime.date): start date of the rolling.
        step_tenor (int): step size of moving in number of days.
        span_tenor (int): span of rolling window in number of days.
    Returns:
        rolling_rtrns (pd.DataFrame): data frame of returns during non-overlapping rolling windows, with 5 columns,
        'StartIdx', 'StartDate', 'EndIdx', 'EndDate', and 'Return'.
    """
    if data_dir is None or not isinstance(data_dir, str) or not os.path.exists(data_dir):
        raise ValueError(f"""Input data_dir is not valid: {data_dir}""")
    if ticker is None or not isinstance(ticker, str):
        raise ValueError(f"""Input ticker is not valid: {ticker}""")
    if start_dt is None or not isinstance(start_dt, datetime.date):
        raise ValueError(f"""Input start_dt is not valid: {start_dt}""")

    filepath = os.path.join(data_dir, f"""{ticker}.csv""")
    if not os.path.exists(filepath):
        raise ValueError(f"""data file for {ticker} does not exist: {filepath}""")

    df = pd.read_csv(filepath)
    if df.shape[0] == 0:
        raise ValueError(f"""data file {filepath} is empty""")

    df['DATE'] = [datetime.datetime.strptime(x, "%Y-%m-%d").date() for x in df['DATE']]
    if start_dt < df['DATE'].iloc[0]:
        raise ValueError(f"""Input start_dt {start_dt} is before the first 'DATE' of input data: {df['DATE'].iloc[0]} """)

    # start the roll by tenor
    step= datetime.timedelta(days=step_tenor)
    span = datetime.timedelta(days=span_tenor)
    period_start_dates = list()
    period_start_idx = list()
    period_end_dates = list()
    period_end_idx = list()
    rtrns = list()
    flag = True
    dt1 = start_dt
    dt2 = None
    while flag:
        # need to consider the case where dt1 and dt2 are not business days
        idx1 = df['DATE'].searchsorted(dt1)
        dt1 = df.loc[idx1, 'DATE']

        dt2 = dt1 + span
        if dt2 > df['DATE'].iloc[-1]:
            break
        idx2 = df['DATE'].searchsorted(dt2)
        dt2 = df.loc[idx2, 'DATE']

        period_start_idx.append(idx1)
        period_end_idx.append(idx2)
        period_start_dates.append(df.loc[idx1, 'DATE'])
        period_end_dates.append(df.loc[idx2, 'DATE'])

        r = math.log(df.loc[idx2, 'CLOSE']) - math.log(df.loc[idx1, 'CLOSE'])
        rtrns.append(r)

        dt1 = dt1 + step

    if len(rtrns) > 0:
        rolling_rtrns = pd.DataFrame({'StartIdx': period_start_idx,
                                      'StartDate': period_start_dates,
                                      'EndIdx': period_end_idx,
                                      'EndDate': period_end_dates,
                                      'Return': rtrns})
    else:
        rolling_rtrns = pd.DataFrame()

    return rolling_rtrns


def calc_tail_risk_from_price(df: pd.DataFrame, period_start_dates: List[datetime.date], period_end_dates: List[datetime.date]) -> float:
    """
    Given lists of pairing period start dates and period end dates, calculate the return over each [period start date, period end date]
    and average all the calculated returns.

    Args:
        df (pandas.DataFrame): data frame that contains column 'DATE' and 'CLOSE'.
        period_start_dates (List[datetime.date]): a list of period start dates.
        period_end_dates (List[datetime.date]): a list of period end dates.

    Returns:
        tail_risk (float): tail risk as the mean of calculated returns.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0 or 'DATE' not in df.columns or 'CLOSE' not in df.columns:
        raise ValueError("Input df is not valid")
    if period_start_dates is None or not isinstance(period_start_dates, list) or len(period_start_dates) == 0:
        raise ValueError(f"""Input start_dates is not valid""")
    if period_end_dates is None or not isinstance(period_end_dates, list) or len(period_end_dates) == 0:
        raise ValueError(f"""Input end_dates is not valid""")
    if len(period_start_dates) != len(period_end_dates):
        raise ValueError("start_dates and end_dates have different lengths")

    if df['DATE'].iloc[0] > period_start_dates[0]:
        raise ValueError(f"""Insufficient data: first date {df['DATE'].iloc[0]} is after the first period start date {period_start_dates[0]}""")
    if df['DATE'].iloc[-1] < period_end_dates[-1]:
        raise ValueError(f"""Insufficient data: last date {df['DATE'].iloc[-1]} is before the last period end date {period_end_dates[-1]}""")

    rtrns = list()
    for i in range(len(period_start_dates)):
        if period_start_dates[i] >= period_end_dates[i]:
            raise ValueError("period start date must be before period end date")
        idx1 = df['DATE'].searchsorted(period_start_dates[i])
        idx2 = df['DATE'].searchsorted(period_end_dates[i])
        rtrns.append(math.log(df.loc[idx2, 'CLOSE']) - math.log(df.loc[idx1, 'CLOSE']))

    return np.mean(rtrns)


def calc_tail_risk_from_rtrn(df: pd.DataFrame, period_start_dates: List[datetime.date], period_end_dates: List[datetime.date]) -> float:
    """
    Given lists of pairing period start dates and period end dates, locate the return over each [period start date, period end date]
    and average all the located returns.

    Args:
        df (pandas.DataFrame): data frame that contains column 'DATE' and 'Return'.
        period_start_dates (List[datetime.date]): a list of period start dates.
        period_end_dates (List[datetime.date]): a list of period end dates.

    Returns:
        tail_risk (float): tail risk as the mean of located returns.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0 or 'DATE' not in df.columns or 'Return' not in df.columns:
        raise ValueError("Input df is not valid")
    if period_start_dates is None or not isinstance(period_start_dates, list) or len(period_start_dates) == 0:
        raise ValueError(f"""Input start_dates is not valid""")
    if period_end_dates is None or not isinstance(period_end_dates, list) or len(period_end_dates) == 0:
        raise ValueError(f"""Input end_dates is not valid""")
    if len(period_start_dates) != len(period_end_dates):
        raise ValueError("start_dates and end_dates have different lengths")

    if df['DATE'].iloc[0] > period_start_dates[0]:
        raise ValueError(f"""Insufficient data: first date {df['DATE'].iloc[0]} is after the first period start date {period_start_dates[0]}""")
    if df['DATE'].iloc[-1] < period_end_dates[-1]:
        raise ValueError(f"""Insufficient data: last date {df['DATE'].iloc[-1]} is before the last period end date {period_end_dates[-1]}""")

    rtrns = list()
    for i in range(len(period_start_dates)):
        if period_start_dates[i] >= period_end_dates[i]:
            raise ValueError("period start date must be before period end date")
        idx1 = df['DATE'].searchsorted(period_start_dates[i])
        idx2 = df['DATE'].searchsorted(period_end_dates[i])
        rtrns.append(df[(idx1+1):(idx2+1)]['Return'].sum())

    return np.mean(rtrns)


def drawdown_analysis(data_dir: str, output_dir: str):
    # single ticker drawdown: SPY
    ticker = 'SPY'

    data = pd.read_csv(os.path.join(data_dir, f"""{ticker}.csv"""))
    data['DATE'] = [datetime.datetime.strptime(x, "%Y-%m-%d").date() for x in data['DATE']]

    prices = data['CLOSE'].to_list()
    print("Prices:", prices[0:3])

    rtrns = [math.log(prices[i]) - math.log(prices[i - 1]) for i in range(1, len(prices))]
    print("Log returns:", rtrns[0:2])

    dd = drawdown(rtrns=rtrns, type='log')
    dd.index = data['DATE']
    dd.to_csv(os.path.join(output_dir, f"""{ticker}_dd.csv"""), index=True)
    print(dd.head())

    prices = data['CLOSE'].to_list()
    rtrns = [math.log(prices[i]) - math.log(prices[i - 1]) for i in range(1, len(prices))]

    dd = drawdown(rtrns=rtrns, type='log')
    dd.index = data['DATE']
    print(dd.head())

    # portfolio drawdown
    tickers = ['AGG', 'BND', 'DBC', 'GSG', 'JNK', 'SPY', 'TIP', 'VBR', 'VNQ', 'VTI', 'VTV', 'VWO']
    start_dt = datetime.date(2007, 12, 11)
    end_dt = datetime.date(2025, 4, 21)
    rtrns_all = calc_rtrns_all(tickers=tickers,
                               data_dir=data_dir,
                               start_dt=start_dt,
                               end_dt=end_dt)
    print(rtrns_all.describe())
    print(rtrns_all.head(3))
    print(rtrns_all.tail(3))

    # portfolio drawdown: Bond20-Stock80
    rtrns_2080 = calc_port_rtrns(rtrns_all, {'BND': 0.2, 'VTI': 0.8})
    print(rtrns_2080.head())

    dd_2080 = drawdown(rtrns=rtrns_2080['Portfolio'].to_list(), type='log')
    dd_2080.index = [start_dt] + list(rtrns_2080.index)
    print(dd_2080.head())

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    axes[0].plot(dd_2080.index, dd_2080['cum_rtrn'], marker='.', linestyle='-')
    axes[0].plot(dd_2080.index, dd_2080['max_cum_rtrn'], marker='.', linestyle='-')
    axes[1].plot(dd_2080.index, dd_2080['drawdown'], marker='.', linestyle='-')

    plt.title(f"""Drawdown Plot of Bond20-STOCK80 Portfolio""")
    plt.xlabel('Date')
    plt.ylabel('Portfolio Return')

    plt.grid(True)

    plt.show()

    # minimal risk portfolio: 50% stock, 30% bond, 20% alternative
    label1 = 'MinimalRisk'
    weights1 = {'AGG': 0.0655, 'BND': 0.15, 'DBC': 0.15, 'GSG': 0.05, 'SPY': 0.15, 'TIP': 0.0845, 'VTI': 0.1125,
                'VTV': 0.15, 'VWO': 0.0875}

    label2 = 'Bond50_Stock50'
    weights2 = {'BND': 0.5, 'VTI': 0.5}

    label3 = 'Bond20_Stock80'
    weights3 = {'BND': 0.2, 'VTI': 0.8}

    label4 = 'Stock100'
    weights4 = {'VTI': 1.0}

    port_rtrns = pd.DataFrame(index=rtrns_all.index)
    port_dd = pd.DataFrame()
    port_dd_dict = dict()
    for l, w in [(label1, weights1), (label2, weights2), (label3, weights3), (label4, weights4)]:
        rtrns = calc_port_rtrns(rtrns_all, w)
        r = rtrns['Portfolio'].to_list()
        port_rtrns[l] = r
        labelled_dd = drawdown(r, 'log')
        labelled_dd.index = [start_dt] + list(rtrns_all.index)
        port_dd[l] = labelled_dd['drawdown']
        port_dd_dict[l] = labelled_dd

    port_dd.index = [start_dt] + list(rtrns_all.index)

    print("\n--- portfolio returns\n\n")
    print(port_rtrns.head())
    print(port_rtrns.tail())
    print(port_rtrns.describe())

    print("\n--- portfolio drawdowns\n\n")
    print(port_dd.head())
    print(port_dd.tail())
    print(port_dd.describe())


if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        data_dir = os.path.join(os.getcwd(), '..', 'data', 'etf')
        output_dir = os.path.join(os.getcwd(), '..', 'output')

        drawdown_analysis(data_dir=data_dir, output_dir=output_dir)

        rolling_rtrns = find_rolling_rtrns(data_dir=data_dir,
                                           ticker="SPY",
                                           start_dt=datetime.date(2005, 1, 4),
                                           num_days=30)

        tail_pct = 0.05,
        threshold = rolling_rtrns['Return'].quantile(tail_pct)
        spy_tail_rtrns = rolling_rtrns[rolling_rtrns['Return'] <= threshold.iloc[0]]
        tail_risk = spy_tail_rtrns['Return'].mean()
        print(spy_tail_rtrns.head())
        print(tail_risk)

        for ticker in ['SPY', 'VTV', 'VBR', 'VTI', 'VWO', 'BND', 'AGG', 'JNK', 'TIP', 'VNQ', 'GSG', 'DBC']:
            filepath = os.path.join(data_dir, f"""{ticker}.csv""")
            df = pd.read_csv(filepath)
            df['DATE'] = [datetime.datetime.strptime(x, "%Y-%m-%d").date() for x in df['DATE']]
            tail_rtrn = calc_tail_risk_from_price(df=df,
                                                  period_start_dates=spy_tail_rtrns['StartDate'].to_list(),
                                                  period_end_dates=spy_tail_rtrns['EndDate'].to_list())
            print(f"""{ticker} tail return: {tail_rtrn}""")

    except Exception as err:
        print("Objective unit port failed: " + str(err))
        print(traceback.format_exc())
