from time import sleep
import ccxt
import pandas as pd
import config
from ta.momentum import rsi

def initiate_exchange():

    # exchange = ccxt.binance({
    # 'enableRateLimit': True,
    # 'apiKey': config.key,
    # 'secret': config.secret
    # })
    exchange = ccxt.phemex()
    exchange.set_sandbox_mode(True)
    exchange.apiKey = config.key
    exchange.secret = config.secret
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
    binance = ccxt.binance()
    df = pd.DataFrame(ccxt.gemini().fetch_ohlcv(symbol=symbol,timeframe=time), columns=columns)
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

    
def trading_bot(exchange,symbol,time, investment, upper_limit, lower_limit ,open_position=False):
    print(open_position)
    price_buy = 0
    price_sell = 0
    while True:
        df = getData(exchange,symbol=symbol,time=time)
        df = RSI(df)
        position = strategy(df, lower_limit,upper_limit)
        print(df.iloc[-1:])

        bitcoin_ticker= exchange.fetch_ticker(symbol)
        price = bitcoin_ticker['close']
        qty = investment/price
        if not open_position:
            if position == 'Buy':
                print('Executing buy order',price)
                
                #exchange.createLimitBuyOrder(symbol=symbol,amount= qty, price =price)
                exchange.create_market_buy_order(symbol = symbol, amount = qty)
                open_position = True 
                timestamp,order_price,order_type,order_qty = get_closed_order(exchange,symbol)
                account_balance = get_balance(exchange)
                price_buy = order_price
                profit = 0
                create_log(timestamp,order_price,order_type,order_qty,account_balance,profit)

                break
            
        
    if open_position:
        while True:
            df = getData(exchange,symbol=symbol,time=time)
            df = RSI(df)
            position = strategy(df, lower_limit,upper_limit)
            print(df.iloc[-1:])
            if position == 'Sell':
                print('Executing sell order')
                exchange.create_market_sell_order(symbol = symbol, amount = qty)
                open_position = False 
                timestamp,order_price,order_type,order_qty = get_closed_order(exchange,symbol)
                account_balance = get_balance(exchange)
                price_sell = order_price
                profit = price_sell - price_buy
                create_log(timestamp,order_price,order_type,order_qty,account_balance,profit)

                break 
            
    
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
    file_path = '/home/jakub/Python-Projects/Gemini-Trading'
    file_name = 'trades.csv'
    df = pd.DataFrame({'DateTime': {},'Order_Price': {}, 'Order_Type': {},
    'Order_Qty': {}, 'Balance': {}, 'Profit': {}
    })

    df = df.append({'DateTime': time,'Order_Price': order_price, 'Order_Type': order_type,
    'Order_Qty': order_qty, 'Balance': account_balance, 'Profit': profit
    },ignore_index=True)
    df.to_csv('{}/{}'.format(file_path,file_name))
    
    f = open('{}/trades.txt'.format(file_path),'a')
    f.write('{} {} {} {} {} {} \n'.format(time,order_price,order_type,
                                        order_qty,account_balance,profit))
    



def main():
    symbol = 'BTC/USDT'
    time = '15m'
    investment = 100
    lower_limit = 70
    upper_limit = 30
    exchange = initiate_exchange()
    
    #get_closed_order(exchange,symbol)
    get_balance(exchange)
    while(True):
        trading_bot(exchange,symbol,time, investment, lower_limit,upper_limit,open_position=False)
        
    
    

main()


