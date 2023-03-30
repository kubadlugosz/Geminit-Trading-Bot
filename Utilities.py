import ccxt
import itertools
import pandas as pd
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
    binance = ccxt.binanceus()
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