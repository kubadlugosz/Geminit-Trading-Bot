import ccxt
import pandas as pd
import ta
from ta.momentum import rsi
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
import itertools
import warnings
from tqdm import tqdm
import itertools
import backtrader as bt
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")


# class Data:
#     def __init__(self,symbol, timeframe):

       
#         #self.exchange = ccxt.get_exchange(exchange_name)
#         self.symbol = symbol
#         self.timeframe = timeframe
#         self.df = None

#     def fetch_data(self):
#         candles = ccxt.gemini().fetch_ohlcv(self.symbol, timeframe=self.timeframe)
#         self.df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
#         self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], unit="ms")
#         self.df.set_index("timestamp", inplace=True)

#     def run(self):
#         self.fetch_data()

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
    df = pd.DataFrame(ccxt.gemini().fetch_ohlcv(symbol=symbol,timeframe=time), columns=columns)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    return df

class Backtester:
    def __init__(self, data, strategy,initial_account_value, investment_per_trade,fee_per_trade):
        self.data = data
        self.strategy = strategy
        self.initial_account_value = initial_account_value
        self.investment_per_trade = investment_per_trade
        self.fee_per_trade = fee_per_trade


    def run_backtest(self, **kwargs):
        params = kwargs
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        # Apply the strategy to the data
        signals = self.strategy.generate_signals(data,**params)

        # Calculate the returns based on the signals and the data
        signals['Returns'] = signals['close'].pct_change() * signals['signal'].shift(1)
        # Remove the first row, which will have a NaN value for returns
        signals = signals.iloc[1:]
        # Calculate the position size based on the investment per trade and account value
        signals['Position Size'] = self.investment_per_trade / signals['close']
        # Calculate the trading fees (assuming 0.1% per trade)
        signals['Fees'] = signals['Position Size'].abs() * self.fee_per_trade
        # Calculate the cash flow for each trade
        signals['Cash Flow'] = -signals['Position Size'].diff() * signals['close'] - signals['Fees']
        # Calculate the cumulative cash flow
        signals['Cumulative Cash Flow'] = signals['Cash Flow'].cumsum()
        # Calculate the account value for each trade
        signals['Account Value'] = self.initial_account_value + signals['Cumulative Cash Flow']
        # Calculate the cumulative returns
        signals['Cumulative Returns'] = (1 + signals['Returns']).cumprod()
        # Calculate the total profit in whole dollars
        total_returns = signals['Cumulative Returns'][-1]
        total_profit = round(self.initial_account_value * self.investment_per_trade * total_returns)
        # Calculate the Sharpe ratio
        sharpe_ratio = (signals['Returns'].mean() / signals['Returns'].std()) * np.sqrt(252)
        # Calculate win rate
        # Calculate the total number of trades
        total_trades = len(signals[signals['signal'] != 0])
        # Calculate the win column
        signals['win'] = signals['Returns'] > 0
        
        winning_trades = len(signals[signals['win']==True])
        # Calculate the win rate as a percentage
        win_rate = 100 * winning_trades / total_trades
        signals.to_csv('signals.csv')
        # Return the results
        return {'sharpe_ratio': sharpe_ratio, 'total_profit': total_profit, 'win_rate': win_rate}
       
    
    def plot_backtest(self, **kwargs):    
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        params = kwargs
        # Apply the strategy to the data
        signals = self.strategy.generate_signals(data,**params)
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
       
       
    
    def optimize_parameters(self, parameter_values):
        results = []
        with tqdm(total=len(parameter_values)) as pbar:
            for params in parameter_values:
                result = self.run_backtest(**params)
                result.update(params)
                results.append(result)
                pbar.update(1)
            
        results_df = pd.DataFrame(results)
        max_sharpe_ratio = results_df['sharpe_ratio'].max()
        max_sharpe_ratio_params = results_df.loc[results_df['sharpe_ratio'].idxmax()].to_dict()
        
        return {'max_sharpe_ratio': max_sharpe_ratio, 'max_sharpe_ratio_params': max_sharpe_ratio_params}
    
    
    
    
    

class MyStrategy:
    def generate_signals(self, data,**params):
        from ta.momentum import rsi
        # Initialize an empty signals Series with the same index as the data
        signals = pd.Series(index=data.index)
        
        # Define your trading signals here
        lower_limit= params['buy_threshold'] 
        upper_limit = params['sell_threshold']
        
        # Calculate the RSI indicator
        data["RSI"] = rsi(data["close"],window=17)
        print(params['rsi_period'])
        #data = rsi_calculation(data,params['rsi_period'])
        print(data)
        #data["RSI"] = ta.momentum.RSIIndicator(close=data["close"], window=params['rsi_period']).rsi()
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
            
       

        data['signal'] = data['signal'].astype("float")
        
     
 
        return data

class Trading_Bot:
    def __init__(self, data, strategy,initial_account_value, investment_per_trade,fee_per_trade):
        self.data = data
        self.strategy = strategy
        self.initial_account_value = initial_account_value
        self.investment_per_trade = investment_per_trade
        self.fee_per_trade = fee_per_trade

    def run_strategy(self, **kwargs):
        params = kwargs
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        # Apply the strategy to the data
        signals = self.strategy.generate_signals(data,**params)
        signal = signals.reset_index().iloc[-1]
        print(signal.timestamp,signal.close,signal.RSI,signal.signal)






def generate_parameter_combinations(param_names, param_ranges):
    param_combinations = itertools.product(*param_ranges)
    parameter_values = [{param_names[i]: p[i] for i in range(len(param_names))} for p in param_combinations]
    return parameter_values


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


def rsi_calculation(df,rsi_period):


    # Define the time period
    period = rsi_period

    # Calculate the price change
    delta = df['close'].diff()

    # Get the positive and negative price changes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and average loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the Relative Strength Index (RSI)
    rsi = 100 - (100 / (1 + rs))

    # Add the RSI values to the DataFrame
    df['RSI'] = rsi

    return df



def main():
    #while True:
        # create a new instance of the Data class for the BTC/USDT symbol on the Binance exchange with a 1-minute timeframe
        data = Data("BTC/USDT", "1m")

        # fetch data for the specified symbol and timeframe
        

        # print the final DataFrame
        #print(data.df.iloc[-1:])
      
        # 
     
    
        # Define the parameter values to optimize over
        #initial_parameter_values = {'rsi_period':14, 'buy_threshold':30, 'sell_threshold':70}
        initial_parameter_values ={'rsi_period': 14.0, 'buy_threshold': 6.0, 'sell_threshold': 60.0}
        param_names = ['rsi_period', 'buy_threshold', 'sell_threshold']
        param_ranges = [[x for x in range(1,20)], [x for x in range(1,30,5)], [x for x in range(60,100,5)]]
        parameter_values = generate_parameter_combinations(param_names, param_ranges)
        
        # Create a new instance of the Backtester class with the data and strategy

        strategy = MyStrategy()
        backtest = Backtester(data.df, strategy,100, 10,0.0099)
        
        # print(backtest.run_backtest(**initial_parameter_values))
        # backtest.plot_backtest(**initial_parameter_values)
        # #Run the parameter optimization
        # results = backtest.optimize_parameters(parameter_values)

        # # # Print the results
        # print(results['max_sharpe_ratio_params'])
        data = Data("BTC/USDT", "1m")
        data.run()
        # while True:
        #     data.run()
        #     trading_bot = Trading_Bot(data.df, strategy,100, 10,0.0099)

        #     # fetch data for the specified symbol and timeframe
        #     trading_bot.run_strategy(**initial_parameter_values)
        #     sleep(1)


        print(rsi_calculation(data.df, 14))


        #print(backtest.run_backtest())
       

main()





