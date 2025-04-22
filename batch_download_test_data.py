# script to download test security prices

import os
import traceback

import yfinance as yf

if __name__ == '__main__':
    """after downloading the data, run script clean_data.sh and change_headers.sh."""
    try:
        data_dir = os.path.join(os.getcwd(), "test", "test_data", "etf")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        #security_list = ['SPY', 'VTV', 'VBR', 'VTI', 'VWO', 'BND', 'AGG', 'JNK', 'TIP', 'VNQ', 'GSG', 'DBC']
        security_list = ['GLD']
        period = '1y'
        interval = '1d'

        for ticker in security_list:
            #df = yf.download(ticker, period=period, interval=interval)
            df = yf.download(ticker, start="2024-01-01", end="2025-04-21")
            df['SEC_ID'] = ticker
            print(df.head())
            df.to_csv(os.path.join(data_dir, ticker + '.csv'))

    except Exception as err:
        print('Data downloading batch failed: ' + str(err))
        print(traceback.format_exc())