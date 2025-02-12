# script to download security prices

import os
import traceback

import yfinance as yf

if __name__ == '__main__':
    """after downloading the data, run script clean_data.sh and change_headers.sh."""
    try:
        data_dir = os.path.join(os.getcwd(), "data", "etf")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        security_list = ['SPY', 'VTV', 'VBR', 'VTI', 'VWO', 'BND', 'AGG', 'JNK', 'TIP', 'VNQ', 'GSG', 'DBC']
        period = '1y'
        interval = '1d'

        for ticker in security_list:
            df = yf.download(ticker, period=period, interval=interval)
            df['SEC_ID'] = ticker
            print(df.head())
            df.to_csv(os.path.join(data_dir, ticker + '.csv'))

    except Exception as err:
        print('Data downloading batch failed: ' + str(err))
        print(traceback.format_exc())