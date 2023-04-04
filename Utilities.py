import ccxt
import itertools
import pandas as pd
import config
import numpy as np
import ta
from scipy import stats
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
    binance = ccxt.gemini()
    # Fetch historical OHLCV data
    ohlcv = binance.fetch_ohlcv(symbol, time)
    
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(ohlcv, columns=columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/Chicago')

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    return df




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

    # Create new DataFrame with the same columns as input
    df_so = df.copy()

    # Extract the parameters from the dictionary
    k_period = params['k_period']
    d_period = params['d_period']

    # Calculate the highest high and lowest low over the past k_period periods
    df_so['HH'] = df_so['High'].rolling(k_period).max()
    df_so['LL'] = df_so['Low'].rolling(k_period).min()

    # Calculate the %K line
    df_so['K'] = 100 * ((df_so['Close'] - df_so['LL']) / (df_so['HH'] - df_so['LL']))

    # Calculate the %D line
    df_so['D'] = df_so['K'].rolling(d_period).mean()
    df_so = df_so.dropna()
    df_so = df_so.reset_index()
    # Return the new DataFrame with %K and %D columns
    return df_so




def calculate_atr_stoploss(df, length=14):
    df['TR'] = np.max([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=0)
    df['ATR'] = df['TR'].rolling(window=length).mean()
    # Set inputs for stop loss calculation
    multiplier = 1.5
    src1 = 'High'
    src2 = 'Low'

    # Calculate stop loss levels
    df['ATR_High'] = df[src1] - multiplier * df['ATR']
    df['ATR_Low'] = df[src2] + multiplier * df['ATR']
    multiplier = 1.5
    src1 = 'High'
    src2 = 'Low'

    # Calculate stop loss levels
    df['ATR_High'] = df[src1] - multiplier * df['ATR']
    df['ATR_Low'] = df[src2] + multiplier * df['ATR']
    
    return df


def linear_regression_channel(df, std_deviation, look_back):
    """
    This function returns a dataframe with prices corresponding to the Upper Deviation,
    Slope, Lower Deviation, and Correlation columns of the Linear Regression Channel.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be used to calculate the linear regression channel.
        
    std_deviation : int
        The number of standard deviations to be used for calculating the channel.
        
    look_back : int
        The number of data points to be used for calculating the linear regression slope.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the prices corresponding to the Upper Deviation, Slope,
        Lower Deviation, and Correlation columns of the Linear Regression Channel.
    """
    # Calculate the linear regression slope
    df['Slope'] = df['Close'].rolling(window=look_back,min_periods=1).apply(lambda x: stats.linregress(range(len(x)), x)[0])
    
    # Calculate the linear regression intercept
    df['Intercept'] = df['Close'].rolling(window=look_back,min_periods=1).apply(lambda x: stats.linregress(range(len(x)), x)[1])
    
    # Calculate the linear regression channel
    df['Upper_Deviation'] = df['Intercept'] + std_deviation * df['Close'].rolling(window = look_back,min_periods=1).std()
    df['Lower_Deviation'] = df['Intercept'] - std_deviation * df['Close'].rolling(window = look_back,min_periods=1).std()
    
    # Calculate the correlation coefficient
    df['Correlation'] = df['Close'].rolling(window=look_back,min_periods=1).corr(df['Slope'])
    
    return df


def calculate_vzo(df, length=14):
    close_prices = df['Close']
    volumes = df['Volume']
    volume_direction = close_prices.diff()
    volume_direction[volume_direction >= 0] = volumes[volume_direction >= 0]
    volume_direction[volume_direction < 0] = -volumes[volume_direction < 0]
    vzo_volume = pd.Series(volume_direction).ewm(span=length, min_periods=length).mean()
    total_volume = pd.Series(volumes).ewm(span=length, min_periods=length).mean()
    vzo = 100 * vzo_volume / total_volume
    df['VZO'] = vzo
    return df



