from time import sleep
import ccxt
import pandas as pd
import config
from ta.momentum import rsi

def initiate_exchange():

    exchange = ccxt.gemini({
    'enableRateLimit': True,
    'apiKey': config.key,
    'secret': config.secret
    })
    exchange.set_sandbox_mode(True)
    return exchange



def getData(exchange,symbol,time):
    """
    1504541580000, // UTC timestamp in milliseconds, integer
    4235.4,        // (O)pen price, float
    4240.6,        // (H)ighest price, float
    4230.0,        // (L)owest price, float
    4230.7,        // (C)losing price, float
    37.72941911    // (V)olume float (usually in terms of the base currency, the exchanges docstring may list whether quote or base units are used)
    """
    columns = ['Date','Open','High Price','Low Price','Close','Volume']
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol=symbol,timeframe=time), columns=columns)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    return df

def RSI(df):
    df['RSI']  = rsi(df.Close,window=14)
    return df

def strategy(df,lower_limit,upper_limit):
    rsi = df['RSI'].iloc[-1]
    if rsi <= lower_limit:
        return 'Buy'
    elif rsi >= upper_limit:
        return 'Sell'
    else:
        return 'Neutral'

    
def execute_order(exchange,symbol,time, qty, open_position=False):
    print(open_position)
    while True:
        df = getData(exchange,symbol=symbol,time=time)
        df = RSI(df)
        position = strategy(df, 47,55)
        print(df.iloc[-1:])
        if not open_position:
            if position == 'Buy':
                print('Executing buy order')

                open_position = True 
                break
            sleep(60)
        
    if open_position:
        while True:
            df = getData(exchange,symbol=symbol,time=time)
            df = RSI(df)
            position = strategy(df, 47,55)
            print(df.iloc[-1:])
            if position == 'Sell':
                print('Executing sell order')
                open_position = False 
                break 
            sleep(60)
    


def main():
    symbol = 'BTC/USD'
    time = '1m'
    exchange = initiate_exchange()
    #print(exchange.fetch_balance())
    while True:
        execute_order(exchange , symbol,time, 1, open_position=False)
        
    
    

main()


