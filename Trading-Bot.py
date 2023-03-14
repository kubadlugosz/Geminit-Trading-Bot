import ccxt
import pandas as pd
import ta
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo

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


class Backtester:
    def __init__(self, data, strategy):
        self.data = data
        self.strategy = strategy
        
    def run_backtest(self):
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        # Apply the strategy to the data
        signals = self.strategy.generate_signals(data)
        print(signals['signal'])
        # Calculate the returns based on the signals and the data
        signals['Returns'] = signals['close'].pct_change() * signals["signal"].shift(1)
        print(signals)
        # Remove the first row, which will have a NaN value for returns
        signals = signals.iloc[1:]
        print(signals)
        # Calculate the cumulative returns
        signals['Cumulative Returns'] = (1 + signals['Returns']).cumprod()
        
        # Calculate the Sharpe ratio
        sharpe_ratio = (signals['Returns'].mean() / signals['Returns'].std()) * np.sqrt(252)
        
        
        # Create a candlestick chart of the data
        candlestick = go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        )

        # Create a scatter plot of the buy and sell signals
        buys = signals[signals['signal'] == 1]
        sells = signals[signals['signal'] == -1]

        buy_scatter = go.Scatter(
            x=buys.index,
            y=buys['close'],
            mode='markers',
            name='Buy',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green'
            )
        )

        sell_scatter = go.Scatter(
            x=sells.index,
            y=sells['close'],
            mode='markers',
            name='Sell',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red'
            )
        )

        # Create a layout for the chart
        layout = go.Layout(
            title='Trading Signals',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price')
        )

        # Combine the candlestick chart and the scatter plots into a single figure
        fig = go.Figure(data=[candlestick, buy_scatter, sell_scatter], layout=layout)

        # Display the figure
        fig.show()
        # Return the results
        return {'returns': signals['Cumulative Returns'][-1], 'sharpe_ratio': sharpe_ratio}
    
    
    
    
    

class MyStrategy:
    def generate_signals(self, data):
        # Initialize an empty signals Series with the same index as the data
        signals = pd.Series(index=data.index)
        
        # Define your trading signals here
        lower_limit= 20
        upper_limit = 50
        
        # Calculate the RSI indicator
        data["RSI"] = ta.momentum.RSIIndicator(close=data["close"], window=14).rsi()
        data = data.dropna()
        
        # # Generate the signals based on the RSI value
        #data["signal"] = data['RSI'].apply(lambda x: 1 if x < lower_limit else -1 if x > upper_limit else 0)
        # Create the 'signal' column
        #data = data.copy()
        data['signal'] = 0
        
        
        # Set a flag to track if we're currently in a position
        in_position = False
        
        # Loop through each row and set the signal based on the RSI and previous position
        
        for i in range(1, len(data)):
            rsi = data['RSI'][i]
            if in_position:
                # If we're currently in a position, continue holding until a Sell signal
                if rsi >= upper_limit:
                    data['signal'][i] = -1
                    in_position = False
                else:
                    data['signal'][i] = 0
            else:
                # If we're not currently in a position, Buy if RSI is below the lower limit
                if rsi <= lower_limit:
                    data['signal'][i] = 1
                    in_position = True
                else:
                    data['signal'][i] = 0
            
            
            
            
            # if data['RSI'][i] < lower_limit and data['signal'][i-1] != 1:
                
            #     data['signal'][i] = 1
            # # Check if RSI is above the high threshold and previous position was not sell
            # elif data['RSI'][i] > upper_limit and data['signal'][i-1] != -1:
            #     data['signal'][i] = -1
            # # Otherwise, hold the previous position
            # else:
            #     data['signal'][i] = 0
       

        data['signal'] = data['signal'].astype("float")
        
        for i in data['signal']:
           print(i)
        
        
        
        return data







def main():
    #while True:
        # create a new instance of the Data class for the BTC/USDT symbol on the Binance exchange with a 1-minute timeframe
        data = Data("BTC/USDT", "15m")

        # fetch data for the specified symbol and timeframe
        data.run()

        # print the final DataFrame
        #print(data.df.iloc[-1:])
      
        # 
     
      
        strategy = MyStrategy()
        backtest = Backtester(data.df, strategy)
        print(backtest.run_backtest())
       

main()





