import ccxt
import itertools
import pandas as pd
import config
import numpy as np
import ta
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from binance.client import Client
import talib
import pandas_ta as ta
def getData(symbol,time):
    """
    1504541580000, // UTC timestamp in milliseconds, integer
    4235.4,        // (O)pen price, float
    4240.6,        // (H)ighest price, float
    4230.0,        // (L)owest price, float
    4230.7,        // (C)losing price, float
    37.72941911    // (V)olume float (usually in terms of the base currency, the exchanges docstring may list whether quote or base units are used)
    """
    
    # Initialize the Binance exchange object
    binance = ccxt.binance()
    # Fetch historical OHLCV data
    #symbol = symbol.replace('USDT', '/USDT')
    ohlcv = binance.fetch_ohlcv(symbol, time)
    
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(ohlcv, columns=columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/Chicago')

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    return df

# def getData(symbol, time):
#     """
#     1504541580000, // UTC timestamp in milliseconds, integer
#     4235.4,        // (O)pen price, float
#     4240.6,        // (H)ighest price, float
#     4230.0,        // (L)owest price, float
#     4230.7,        // (C)losing price, float
#     37.72941911    // (V)olume float (usually in terms of the base currency, the exchanges docstring may list whether quote or base units are used)
#     """
    
#     # Initialize the Binance client object
#     client = Client(api_key=config.key,api_secret=config.secret,tld='us',testnet=True)
#     # Fetch historical klines data
#     klines = client.get_historical_klines(symbol, time)

#     # Convert klines data to dataframe
#     columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
#                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
#                'taker_buy_quote_asset_volume', 'ignore']
#     df = pd.DataFrame(klines, columns=columns)

#     # Drop unnecessary columns
#     df.drop(['close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
#              'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)

#     # Convert timestamp from milliseconds to datetime
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
#     df['timestamp'] = df['timestamp'].dt.tz_convert('America/Chicago')

#     # Reorder and rename columns
#     df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
#     df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

#     return df


def generate_parameter_combinations(param_names, param_ranges):
    param_combinations = itertools.product(*param_ranges)
    parameter_values = [{param_names[i]: p[i] for i in range(len(param_names))} for p in param_combinations]
    return parameter_values



def initiate_exchange():

    exchange = ccxt.phemex()
    exchange.set_sandbox_mode(True)
    exchange.apiKey = config.key
    exchange.secret = config.secret
    return exchange

def get_closed_order(exchange,symbol):

    order = exchange.fetch_closed_orders(symbol)#[-1]
    print(order)
    time = order['datetime']
    order_type =order['side']
    order_price = order['price']
    order_qty = order['amount']

    return time,order_price,order_type,order_qty

def get_balance(exchange):

    balance = exchange.fetch_balance()
    return balance['USDT']['total']


def create_log(time,order_price,order_type,order_qty,account_balance,profit):
    file_name = 'trades.csv'
    df = pd.DataFrame({'DateTime': {},'Order_Price': {}, 'Order_Type': {},
    'Order_Qty': {}, 'Balance': {}, 'Profit': {}
    })

    df = df.append({'DateTime': time,'Order_Price': order_price, 'Order_Type': order_type,
    'Order_Qty': order_qty, 'Balance': account_balance, 'Profit': profit
    },ignore_index=True)
    df.to_csv(file_name)
    
    f = open('trades.txt','a')
    f.write('{} {} {} {} {} {} \n'.format(time,order_price,order_type,
                                        order_qty,account_balance,profit))
    

def calculate_macd(data, **params):
    """
    Calculates the MACD line, signal line, and zero line for a given DataFrame of stock data.
    :param data: DataFrame containing stock data with at least a 'Close' column and a DatetimeIndex.
    :param EMA Long Period: The period length for the long EMA. Default is 26.
    :param EMA Short Period: The period length for the short EMA. Default is 12.
    :param Signal Line Period: The period length for the signal line. Default is 9.
    :return: DataFrame containing the MACD line, signal line, and zero line as columns, with the same index as the input DataFrame.
    """
    ema_long = data['Close'].ewm(span=params['EMA Long Period'], adjust=False).mean()
    ema_short = data['Close'].ewm(span=params['EMA Short Period'], adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=params['Signal Line Period'], adjust=False).mean()
    zero_line = pd.Series(0, index=data.index)
    macd_df = pd.concat([macd_line, signal_line, zero_line], axis=1)
    macd_df.columns = ['MACD_line', 'Signal_line', 'Zero_line']
    return macd_df



def fibonacci_retracement(df):
    """
    Calculates Fibonacci retracement levels based on the most recent upward or downward trend in a DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns "Open", "High", "Low", and "Close".

    Returns:
        pandas.DataFrame: New DataFrame with additional columns for Fibonacci retracement levels.

    """

    # Create new DataFrame with the same columns as input
    df_fib = df.copy()

    # Calculate the high and low of the most recent trend
    last_trend_high = df_fib.iloc[0]['High']
    last_trend_low = df_fib.iloc[0]['Low']
    last_trend_up = True

    # Loop through each row in the DataFrame and calculate Fibonacci levels
    for i, row in df_fib.iterrows():

        # If the current high is higher than the last trend high, we are in an upward trend
        if row['High'] > last_trend_high:
            last_trend_high = row['High']
            last_trend_low = row['Low']
            last_trend_up = True

        # If the current low is lower than the last trend low, we are in a downward trend
        elif row['Low'] < last_trend_low:
            last_trend_low = row['Low']
            last_trend_high = row['High']
            last_trend_up = False

        # Calculate the retracement levels based on the most recent trend
        if last_trend_up:
            diff = last_trend_high - last_trend_low
            df_fib.at[i, '38.2%'] = last_trend_high - (0.382 * diff)
            df_fib.at[i, '50.0%'] = last_trend_high - (0.5 * diff)
            df_fib.at[i, '61.8%'] = last_trend_high - (0.618 * diff)
        else:
            diff = last_trend_high - last_trend_low
            df_fib.at[i, '38.2%'] = last_trend_low + (0.382 * diff)
            df_fib.at[i, '50.0%'] = last_trend_low + (0.5 * diff)
            df_fib.at[i, '61.8%'] = last_trend_low + (0.618 * diff)

    # Return the new DataFrame with Fibonacci retracement levels
    return df_fib




def crossover(df, s1, s2):
    """
    Calculates the crossover signals between two time-series data.

    Args:
        df (pandas.DataFrame): DataFrame containing the input data.
        s1 (pandas.Series): First time-series data.
        s2 (pandas.Series): Second time-series data.

    Returns:
        pandas.DataFrame: DataFrame with an additional column 'Crossover' with values of 1, 0, -1 indicating the occurrence of a bullish, no crossover, or bearish signal, respectively.

    """
    # Create a new column for crossover signals
    
    df['Crossover'] = 0

    # Loop through the data
    for i in range(1, len(df)):
        if s1[i] > s2[i] and s1[i - 1] <= s2[i - 1]:
            # Bullish crossover signal
            df.loc[i, 'Crossover'] = 1
        elif s1[i] < s2[i] and s1[i - 1] >= s2[i - 1]:
            # Bearish crossover signal
            df.loc[i, 'Crossover'] = -1

    return df


def stochastic_oscillator(df, **params):
    """
    Calculates the Stochastic Oscillator indicator for a DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns "Open", "High", "Low", and "Close".
        params (dict): Dictionary containing the parameter names and values.

    Returns:
        pandas.DataFrame: New DataFrame with additional columns for the %K and %D lines of the Stochastic Oscillator.

    """
    k_period = params['k_period']
    d_period = params['d_period']
    df[['K','D']]=df.ta.stoch(high='High', low='Low',close='Close', k=k_period, d_period=3)
    # # Create new DataFrame with the same columns as input
    # df_so = df.copy()

    # # Extract the parameters from the dictionary
    # k_period = params['k_period']
    # d_period = params['d_period']

    # # Calculate the highest high and lowest low over the past k_period periods
    # df_so['HH'] = df_so['High'].rolling(k_period).max()
    # df_so['LL'] = df_so['Low'].rolling(k_period).min()

    # # Calculate the %K line
    # df_so['K'] = 100 * ((df_so['Close'] - df_so['LL']) / (df_so['HH'] - df_so['LL']))

    # # Calculate the %D line
    # df_so['D'] = df_so['K'].rolling(d_period).mean()
    #df_so = df_so.dropna()
    #df_so = df_so.reset_index(drop=True)
    # Return the new DataFrame with %K and %D columns
    return df




def calculate_atr_stoploss(df, length=14):
   
    multiplier = 1.5

    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=length)
    df['ATR_High'] = df['ATR']  * multiplier + df['High']
    df['ATR_Low'] = df['Low'] - df['ATR']  * multiplier
    return df

def profit_stoploss(df, method):
    df['Take_Profit'] =  0
    df['Stop_Loss']= 0
    if method == 'atr':
        df = calculate_atr_stoploss(df)
        df = df.dropna()
        df = df.reset_index(drop=True)
        for i in range(1, len(df)):
            signal = df['Signal'][i]
            take_profit = df['Middle_Channel'][i]
            atr_high = df['ATR_High'][i]
            atr_low = df['ATR_Low'][i]
            fib_low = df['Fib_0.236'][i]
            fib_high = df['Fib_0.786'][i]
            trend = df['Trend'][i]
            #condition for buy signal
            if signal == 1 and trend == 'Upward trend':
                df['Take_Profit'][i] = take_profit
                df['Stop_Loss'][i] = atr_low
            elif signal == 1 and trend == 'Downward trend':
                df['Take_Profit'][i] = fib_low
                df['Stop_Loss'][i] = atr_low    
            elif signal == -1 and trend == 'Downward trend':
                df['Take_Profit'][i] = take_profit
                df['Stop_Loss'][i] = atr_high
            elif signal == -1 and trend == 'Upward trend':
                df['Take_Profit'][i] = fib_high
                df['Stop_Loss'][i] = atr_high
            else:
                df['Take_Profit'][i] = 0
                df['Stop_Loss'][i] = 0
        return df
    
def linear_regression_channel(df, lookback, std_deviation):
    if 'Close' not in df:
        raise KeyError('Close column not found in input DataFrame')
    if len(df) < lookback:
        raise ValueError('Input DataFrame is too small for the specified lookback period')

    # Compute the linear regression line for each window of size 'lookback'
    x = np.arange(lookback).reshape(-1, 1)
    linreg = LinearRegression()
    linreg_line = df['Close'].rolling(lookback).apply(lambda y: linreg.fit(x, y).intercept_ + linreg.fit(x, y).coef_[0] * x.flatten(), raw=False)
    
    # Compute the Pearson correlations between the closing prices and the linear regression line
    #corr, _ = pearsonr(df['Close'], linreg_line)
    # Compute the upper and lower channels using 'std_deviation' standard deviations
    #std = np.std(df['Close'][-lookback:])
    std = (df['Close'] - linreg_line).rolling(lookback).std()
    upper_channel = linreg_line + std_deviation * std
    lower_channel = linreg_line - std_deviation * std
    
    # Compute the Fibonacci levels
    fib_levels = [0.236, 0.382, 0.618, 0.786]
    fib_values = np.array([(lower_channel + level * (upper_channel - lower_channel)) for level in fib_levels]).T
    
    # Add the upper, middle, and lower channels to the original dataframe as new columns
    df['Upper_Channel'] = upper_channel
    df['Middle_Channel'] = linreg_line
    df['Lower_Channel'] = lower_channel
    
    # Add Fibonacci levels to dataframe
    for i, level in enumerate(fib_levels):
        df[f'Fib_{level:.3f}'] = fib_values[:, i]
    
    # Check the trend of the linear regression line
    trend_is_up = linreg_line.diff() > 0
    trend_is_down = linreg_line.diff() < 0
    df.loc[trend_is_up, 'Trend'] = "Upward trend"
    df.loc[trend_is_down, 'Trend'] = "Downward trend"
    df.loc[~(trend_is_up | trend_is_down), 'Trend'] = "Sideways trend"
    
    return df

def linear_regression_channel(df, lookback, std_deviation):
    if 'Close' not in df:
        raise KeyError('Close column not found in input DataFrame')
    if len(df) < lookback:
        raise ValueError('Input DataFrame is too small for the specified lookback period')
    channel_df = pd.DataFrame()
    for i in range(0, len(df), lookback):
        subset_df = df.iloc[i:i+lookback]
        # Compute the linear regression line for each window of size 'lookback'
        x = np.arange(lookback).reshape(-1, 1)
        linreg = LinearRegression()
        linreg.fit(x, subset_df['Close'])
        slope = linreg.coef_[0]
        intercept = linreg.intercept_
        linreg_line = intercept + slope * x.flatten()
        
        # Check the trend of the linear regression line
        trend_is_up =  slope > 0
        trend_is_down = slope < 0
        # Compute the Pearson correlations between the closing prices and the linear regression line
        #corr, _ = pearsonr(df['Close'], linreg_line)
        # Compute the upper and lower channels using 'std_deviation' standard deviations
        #std = np.std(df['Close'][-lookback:])
        std = np.std(subset_df['Close']-linreg_line)
        upper_channel = linreg_line + std_deviation * std
        lower_channel = linreg_line - std_deviation * std
        
        subset_df['Fib_0.236'] = lower_channel + 0.236 * (upper_channel - lower_channel)
        subset_df['Fib_0.382'] = lower_channel + 0.382 * (upper_channel - lower_channel)
        subset_df['Fib_0.618'] = lower_channel + 0.618 * (upper_channel - lower_channel)
        subset_df['Fib_0.786'] = lower_channel + 0.786 * (upper_channel - lower_channel)
        # Add the upper, middle, and lower channels to the original dataframe as new columns
        subset_df['Upper_Channel'] = upper_channel
        subset_df['Middle_Channel'] = linreg_line
        subset_df['Lower_Channel'] = lower_channel

        subset_df['Trend'] = "No clear trend"
        if slope > 0:
            subset_df['Trend'] = "Upward trend"
        elif slope < 0:
            subset_df['Trend'] = "Downward trend"
        elif slope == 0:
            subset_df['Trend'] = "Sideways trend"
        #df['Correlation'] = corr
        channel_df = channel_df.append(subset_df, ignore_index=True)
    
    
    return channel_df



def calculate_vzo(df, **params):
    vzo_len = params['vzo_length']
    vzo_smooth = params['vzo_smooth_length']
    svol = np.where(df['Open'] < df['Close'], df['Volume'], -df['Volume'])
    vp = pd.Series(svol).ewm(span=vzo_len).mean()
    tv = df['Volume'].ewm(span=vzo_len).mean()
    vzo_val = 100 * vp / tv
    vzo_val = vzo_val.ewm(span=vzo_smooth).mean()
    df['VZO'] = vzo_val
    return df








