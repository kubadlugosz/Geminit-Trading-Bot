import ccxt
import pandas as pd
import ta
from time import sleep
class Data:
    def __init__(self,symbol, timeframe):

       
        #self.exchange = ccxt.get_exchange(exchange_name)
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = None

    def fetch_data(self):
        candles = ccxt.gemini().fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        self.df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], unit="ms")
        self.df.set_index("timestamp", inplace=True)

    def run(self):
        self.fetch_data()


class Trading_Bot:
    def __init__(self, data):
        self.data = data
        self.positions = []
        self.pnl = []

    def run_strategy(self, strategy):
        print(strategy(self.data))
        last_signal = "sell"  # Set an initial signal to sell to start the cycle
        open_position = False  # For the sake of demonstration, assume there is an open position

        print(open_position)
        price_buy = 0
        price_sell = 0
        #while True:

        position = strategy(self.data)
        print(position)
        print(self.data.iloc[-1:])
     
        #     bitcoin_ticker= exchange.fetch_ticker(symbol)
        #     price = bitcoin_ticker['close']
        #     qty = 100/price
        #     if not open_position:
        #         if position == 'Buy':
        #             print('Executing buy order',price)
                    
        #             #exchange.createLimitBuyOrder(symbol=symbol,amount= qty, price =price)
        #             exchange.create_market_buy_order(symbol = symbol, amount = qty)
        #             open_position = True 
        #             timestamp,order_price,order_type,order_qty = get_closed_order(exchange,symbol)
        #             account_balance = get_balance(exchange)
        #             price_buy = order_price
        #             profit = 0
        #             # create_log(timestamp,order_price,order_type,order_qty,account_balance,profit)

        #             break
                
            
        # if open_position:
        #     while True:
        #         df = getData(exchange,symbol=symbol,time=time)
        #         df = RSI(df)
        #         position = strategy(df, lower_limit,upper_limit)
        #         print(df.iloc[-1:])
        #         if position == 'Sell':
        #             print('Executing sell order')
        #             exchange.create_market_sell_order(symbol = symbol, amount = qty)
        #             open_position = False 
        #             timestamp,order_price,order_type,order_qty = get_closed_order(exchange,symbol)
        #             account_balance = get_balance(exchange)
        #             price_sell = order_price
        #             profit = price_sell - price_buy
        #             create_log(timestamp,order_price,order_type,order_qty,account_balance,profit)

        #             break 


def rsi_strategy(df):
    lower_limit = 20
    upper_limit = 80
    df["RSI"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    rsi = df['RSI'].iloc[-1]
    if rsi <= lower_limit:
        return 'Buy'
    elif rsi >= upper_limit:
        return 'Sell'
    else:
        return 'Neutral'


def main():
    while True:
        # create a new instance of the Data class for the BTC/USDT symbol on the Binance exchange with a 1-minute timeframe
        data = Data("BTC/USDT", "1m")

        # fetch data for the specified symbol and timeframe
        data.run()

        # print the final DataFrame
        print(data.df.iloc[-1:])

        # create a new instance of the Trading_Bot class
        trading_bot = Trading_Bot(data.df)

        # run the trading strategy
       
            
        trading_bot.run_strategy(rsi_strategy)
        sleep(60)
        # pnl = trading_bot.get_pnl()
        # print("Total PNL:", pnl)


main()





