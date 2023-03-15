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
from plotly.subplots import make_subplots
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

def getData(symbol,time):
    """
    1504541580000, // UTC timestamp in milliseconds, integer
    4235.4,        // (O)pen price, float
    4240.6,        // (H)ighest price, float
    4230.0,        // (L)owest price, float
    4230.7,        // (C)losing price, float
    37.72941911    // (V)olume float (usually in terms of the base currency, the exchanges docstring may list whether quote or base units are used)
    """
    columns = ['Date','Open','High','Low','Close','Volume']
    df = pd.DataFrame(ccxt.gemini().fetch_ohlcv(symbol=symbol,timeframe=time), columns=columns)
   
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Date'] = df['Date'].dt.tz_localize('UTC')
    df['Date'] = df['Date'].dt.tz_convert('America/Chicago')
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
        signals = self.strategy.generate_signals_backtest(data,**params)

        # Calculate the returns based on the signals and the data
        signals['Returns'] = signals['Close'].pct_change() * signals['Signal'].shift(1)
        # Remove the first row, which will have a NaN value for returns
        signals = signals.iloc[1:]
        # Calculate the position size based on the investment per trade and account value
        signals['Position Size'] = self.investment_per_trade / signals['Close']
        # Calculate the trading fees (assuming 0.1% per trade)
        signals['Fees'] = signals['Position Size'].abs() * self.fee_per_trade
        # Calculate the cash flow for each trade
        signals['Cash Flow'] = -signals['Position Size'].diff() * signals['Close'] - signals['Fees']
        # Calculate the cumulative cash flow
        signals['Cumulative Cash Flow'] = signals['Cash Flow'].cumsum()
        # Calculate the account value for each trade
        signals['Account Value'] = self.initial_account_value + signals['Cumulative Cash Flow']
        # Calculate the cumulative returns
        signals['Cumulative Returns'] = (1 + signals['Returns']).cumprod()
        signals = signals.dropna().reset_index(drop=True)
        # Calculate the total profit in whole dollars
        total_returns = signals['Cumulative Returns'].iloc[-1] #['Cumulative Returns'][-1]
        total_profit = round(self.initial_account_value * self.investment_per_trade * total_returns)
        # Calculate the Sharpe ratio
        sharpe_ratio = (signals['Returns'].mean() / signals['Returns'].std()) * np.sqrt(252)
        # Calculate win rate
        # Calculate the total number of trades
        total_trades = len(signals[signals['Signal'] != 0])
        # Calculate the win column
        signals['win'] = signals['Returns'] > 0
        
        winning_trades = len(signals[signals['win']==True])
        # Calculate the win rate as a percentage
        win_rate = 100 * winning_trades / total_trades
        signals.to_csv('signals.csv')
        # Return the results
        return {'sharpe_ratio': sharpe_ratio, 'total_profit': total_profit, 'win_rate': win_rate}
       
    
    def plot_backtest(self,indicator=None, **kwargs):    
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        params = kwargs
        # Apply the strategy to the data
        signals = self.strategy.generate_signals_backtest(data,**params)
        # Create a candlestick chart of the data
        candlestick = go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )

        # Create a scatter plot of the buy and sell signals
        buys = signals[signals['Signal'] == 1]
        sells = signals[signals['Signal'] == -1]

        buy_scatter = go.Scatter(
            x=buys['Date'],
            y=buys['Close'],
            mode='markers',
            name='Buy',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green'
            )
        )

        sell_scatter = go.Scatter(
            x=sells['Date'],
            y=sells['Close'],
            mode='markers',
            name='Sell',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='red'
            )
        )
        
        # Create a layout for the chart with a secondary axis for the RSI plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # Add the candlestick chart to the first row of the subplot
        fig.add_trace(candlestick, row=1, col=1)

        # Add the buy and sell signals to the first row of the subplot
        fig.add_trace(buy_scatter, row=1, col=1)
        fig.add_trace(sell_scatter, row=1, col=1)

        # Create a plot of the RSI indicator, if provided, and add it to the second row of the subplot
        if indicator is not None:
            rsi_trace = go.Scatter(
                x=data['Date'],
                y=indicator,
                name='RSI'
            )
            fig.add_trace(rsi_trace, row=2, col=1)

        # Update the layout to include a title and axis labels
        fig.update_layout(
            title='Trading Signals',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price', domain=[0.2, 1]),
            yaxis2=dict(title='RSI', domain=[0, 0.15])
        )

        # Display the figure
        fig.show()
       
       
    
    def optimize_parameters(self, parameter_values):
        results = []
        with tqdm(total=len(parameter_values)) as pbar:
            for params in parameter_values:
                try:
                    result = self.run_backtest(**params)
                    result.update(params)
                    results.append(result)
                    pbar.update(1)
                except:
                    continue
            
        results_df = pd.DataFrame(results)
        max_sharpe_ratio = results_df['sharpe_ratio'].max()
        max_sharpe_ratio_params = results_df.loc[results_df['sharpe_ratio'].idxmax()].to_dict()
        
        return {'max_sharpe_ratio': max_sharpe_ratio, 'max_sharpe_ratio_params': max_sharpe_ratio_params}
    
    
    
    
    

class MyStrategy:
    
    def generate_signals_backtest(self, data,**params):
        from ta.momentum import rsi
        # Initialize an empty signals Series with the same index as the data
        signals = pd.Series(index=data.index)
        
        # Define your trading signals here
        lower_limit= params['buy_threshold'] 
        upper_limit = params['sell_threshold']
        
        # Calculate the RSI indicator
        data["RSI"] = rsi(data["Close"],window=params['rsi_period'])
        
        #data = rsi_calculation(data,params['rsi_period'])

        #data["RSI"] = ta.momentum.RSIIndicator(close=data["close"], window=params['rsi_period']).rsi()
        data = data.dropna()
        data = data.reset_index(drop=True)
        # # Generate the signals based on the RSI value
        #data["signal"] = data['RSI'].apply(lambda x: 1 if x < lower_limit else -1 if x > upper_limit else 0)
        # Create the 'signal' column
        #data = data.copy()
        data['Signal'] = 0
        
        
        # Set a flag to track if we're currently in a position
        in_position = False
        
        # Loop through each row and set the signal based on the RSI and previous position
      
        for i in range(1, len(data)):
            rsi_value = data['RSI'][i]
           
            if in_position:
                # If we're currently in a position, continue holding until a Sell signal
                if rsi_value >= upper_limit:
                    data['Signal'][i] = -1
                    in_position = False
                else:
                    data['Signal'][i] = 0
            else:
                # If we're not currently in a position, Buy if RSI is below the lower limit
                if rsi_value <= lower_limit:
                    data['Signal'][i] = 1
                    in_position = True
                else:
                    data['Signal'][i] = 0
            
       

        data['Signal'] = data['Signal'].astype("float")
        
     
 
        return data
    
    def generate_signals_trading_bot(self, data,**params):
        rsi = data['RSI'].iloc[-1]
        if rsi <= params['buy_threshold'] :
            return 'Buy'
        elif rsi >= params['sell_threshold'] :
            return 'Sell'
        else:
            return 'Neutral'
        
        
        
        

class Trading_Bot:
    def __init__(self, data, strategy,initial_account_value, investment_per_trade,fee_per_trade):
        self.data = data
        self.strategy = strategy
        self.initial_account_value = initial_account_value
        self.investment_per_trade = investment_per_trade
        self.fee_per_trade = fee_per_trade

    def run_strategy(self, **kwargs):
        params = kwargs
        
        open_position=False
        while True:
            # Apply the strategy to the data
            data = self.strategy.generate_signals_backtest(self.data,**params)
            signal = self.strategy.generate_signals_trading_bot(self.data,**params)
            data = data.iloc[-1]
            
            
            ticker= ccxt.gemini().fetch_ticker('ETH/USDT')
            price = ticker['close']
            qty = self.investment_per_trade/price
           
            print(data.Date,data.Close,data.RSI)
            if not open_position:
                if signal == 'Buy':
                    print('Executing buy order',price)
                    open_position = True 
                    
                    
                    break

        if open_position:
            while True:
                if signal == 'Sell':
                    print('Executing sell order',price)
                    open_position = False 




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




def main():
    
    inital_parameters = {'rsi_period': 17, 'buy_threshold': 21.0, 'sell_threshold': 95.0}
    strategy = MyStrategy()
    # backtester = Backtester(df, strategy,100, 10, 0.0099)
    # print(backtester.run_backtest(**inital_parameters))
    # rsi_data = strategy.generate_signals_backtest(df,**inital_parameters)
    # # print(rsi_data)
    # backtester.plot_backtest(rsi_data['RSI'],**inital_parameters)
   
    # param_names = ['rsi_period','buy_threshold','sell_threshold']
    # param_ranges = [{x for x in range(1,20,1)},{x for x in range(1,30,5)},{x for x in range(60,100,5)}]
    # param_combo = generate_parameter_combinations(param_names, param_ranges)
    # print(backtester.optimize_parameters(param_combo))
    
    while True:
        df = getData("ETH/USDT",'1m')  
        trade_bot = Trading_Bot(df, strategy,100, 10,0.0099)
        trade_bot.run_strategy(**inital_parameters)
        sleep(1)

main()





