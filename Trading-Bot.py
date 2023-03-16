import ccxt
import config
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
    def __init__(self, data, strategy,initial_account_value, ivestment_amount,fee_per_trade):
        self.data = data
        self.strategy = strategy
        self.initial_account_value = initial_account_value
        self.ivestment_amount = ivestment_amount
        self.fee_per_trade = fee_per_trade


    def run_backtest(self, **kwargs):
        params = kwargs
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        # Apply the strategy to the data
        signals = self.strategy.generate_signals_backtest(data,**params)
        # Calculate the total profit in whole dollars
        total_profit,account_value,signals= self.calculate_total_profits(signals,self.initial_account_value,self.ivestment_amount,self.fee_per_trade)
        
        # Calculate sharpe ratio
        sharpe_ratio = self.calculate_sharpe_ratio(signals)
       
        # Calculate win rate
        win_rate = self.calculate_win_rate(signals)
        signals.to_csv('signals.csv')
        # Return the results
        return {'sharpe_ratio': sharpe_ratio, 'innitial_account_value':self.initial_account_value, 'account_value': account_value,'total_profit': total_profit, 'win_rate': win_rate}
      
        

    def calculate_total_profits(self, data, initial_account_value, investment_amount, fee_per_trade):
        """
        Calculates the total profits from a trading strategy given a DataFrame with closing prices and trade signals.

        Args:
        - data: A Pandas DataFrame with columns "close" (closing prices) and "signal" (trade signals, where 1 represents a buy signal, 0 represents a hold signal, and -1 represents a sell signal).
        - initial_account_value: The total amount of money in the account at the start of the trading period.
        - dollar_amount_spent_per_trade: The dollar amount spent on each trade.
        - fee_per_trade: The fee associated with each trade.

        Returns:
        - The total profits (or losses) from the trading strategy.
        """
        account_value = initial_account_value
        num_shares = 0
        num_trades = 0
        total_profit = 0
        data['Account Value'] = 0
        data['Gains/Losses'] = 0
        for i in range(len(data)):
            if data['Signal'][i] == 1: # Buy signal
                # Calculate how many tokens were purchased with the investment amount, factoring in the trading fee
                num_shares = (investment_amount - (fee_per_trade*100)) / data['Close'][i]
                # Deduct the total investment amount and trading fee from the account value
                account_value -= investment_amount 
                data['Account Value'][i] = account_value
            elif data['Signal'][i] == -1: # Sell signal
                # Calculate the gross proceeds from selling the tokens, factoring in the trading fee
                gross_proceeds = num_shares * data['Close'][i] - (fee_per_trade*100)
                # Calculate the profit or loss from the trade
                profit_loss = gross_proceeds - investment_amount
                # Add the profit or loss to the total profit
                total_profit += profit_loss
                # Add the gross proceeds (minus trading fee) to the account value
                account_value += gross_proceeds
                # Reset the number of shares to 0
                num_shares = 0
                # Increment the number of trades counter
                num_trades += 1

                data['Account Value'][i] = account_value
                data['Gains/Losses'][i] = profit_loss
            else: # Hold signal
                data['Account Value'][i] = account_value

        # Add the final account value to the total profit
        total_profit += account_value - initial_account_value

        
        
        return total_profit,account_value,data
    
    def calculate_sharpe_ratio(self,data):
        """
        Calculate the Sharpe Ratio from a DataFrame of daily closing prices.

        Parameters:
        df_prices (pandas.DataFrame): A DataFrame of daily closing prices.

        Returns:
        float: The Sharpe Ratio.
        """

        # Calculate the daily returns based on the trade signals
        data['Returns'] = data['Signal'] * (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)

        # Calculate the Sharpe ratio
        R_p = data['Returns'].mean() * 252
        R_f = 0.02 # Assume a risk-free rate of 2%
        sigma_p = data['Returns'].std() * np.sqrt(252)
        sharpe_ratio = (R_p - R_f) / sigma_p


        return sharpe_ratio
    
    def calculate_win_rate(self,data):
        """
        Calculate the win rate for a trading strategy based on a DataFrame of closing prices and signals.

        Parameters:
        data (pandas.DataFrame): A DataFrame containing closing prices and signals.

        Returns:
        float: The win rate as a percentage.
        """
        # Calculate the number of winning and losing trades
        num_wins = len(data[data['Gains/Losses'] > 0])
        num_losses = len(data[data['Gains/Losses'] < 0])

        # Calculate the win rate as a percentage
        win_rate = num_wins / (num_wins + num_losses) * 100

        return win_rate
    
    def plot_backtest(self,indicator=None, **params):    
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        
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
            xaxis=dict(title='Date', tickformat='%Y-%m-%d %H:%M:%S'),
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
        max_win_ratio = results_df['win_rate'].max()
        max_win_ratio_params = results_df.loc[results_df['win_rate'].idxmax()].to_dict()
        
        return {'max_sharpe_ratio': max_win_ratio, 'max_sharpe_ratio_params': max_win_ratio_params}
    
    
    
    
    

class MyStrategy:
    
    def generate_signals_backtest(self, data,**params):
        # from ta.momentum import rsi
        # # Initialize an empty signals Series with the same index as the data
        # signals = pd.Series(index=data.index)
        
        # # Define your trading signals here
        # lower_limit= params['buy_threshold'] 
        # upper_limit = params['sell_threshold']
        
        # # Calculate the RSI indicator
        # #data["RSI"] = rsi(data["Close"],window=params['rsi_period'])
        # data['RSI']= ta.momentum.RSIIndicator(data["Close"], window=params['rsi_period']).rsi()
        # print(data)
        # #data = rsi_calculation(data,params['rsi_period'])

        # #data["RSI"] = ta.momentum.RSIIndicator(close=data["close"], window=params['rsi_period']).rsi()
        # data = data.dropna()
        # data = data.reset_index(drop=True)
        # # # Generate the signals based on the RSI value
        # #data["signal"] = data['RSI'].apply(lambda x: 1 if x < lower_limit else -1 if x > upper_limit else 0)
        # # Create the 'signal' column
        # #data = data.copy()
        # data['Signal'] = 0
        
        
        # # Set a flag to track if we're currently in a position
        # in_position = False
        
        # # Loop through each row and set the signal based on the RSI and previous position
      
        # for i in range(1, len(data)):
        #     rsi_value = data['RSI'][i]
           
        #     if in_position:
        #         # If we're currently in a position, continue holding until a Sell signal
        #         if rsi_value >= upper_limit:
        #             data['Signal'][i] = -1
        #             in_position = False
        #         else:
        #             data['Signal'][i] = 0
        #     else:
        #         # If we're not currently in a position, Buy if RSI is below the lower limit
        #         if rsi_value <= lower_limit:
        #             data['Signal'][i] = 1
        #             in_position = True
        #         else:
        #             data['Signal'][i] = 0
            
       

        # data['Signal'] = data['Signal'].astype("float")
        


        #MACD trading strategy
        macd_df = calculate_macd(data, **params)
        data = pd.concat([data, macd_df], axis=1)
        # Set a flag to track if we're currently in a position
        in_position = False
        
        # Loop through each row and set the signal based on the RSI and previous position
        data['Signal'] = 0
        for i in range(1, len(data)):
            macd = data['MACD_line'][i]
            zero_line = data['Zero_line'][i]
            signal_line = data['Signal_line'][i]
           
            if in_position:
                # If we're currently in a position, continue holding until a Sell signal
                if (macd < signal_line) & (macd > zero_line) & (signal_line > zero_line): 
                    data['Signal'][i] = -1
                    in_position = False
                else:
                    data['Signal'][i] = 0
            else:
                # If we're not currently in a position, Buy if RSI is below the lower limit
                if (macd > signal_line) & (macd < zero_line) & (signal_line < zero_line):
                    data['Signal'][i] = 1
                    in_position = True
                else:
                    data['Signal'][i] = 0

      

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
    def __init__(self, data,exchange,symbol,strategy,initial_account_value, ivestment_amount,fee_per_trade):
        self.data = data
        self.exchange = exchange
        self.symbol = symbol
        self.strategy = strategy
        self.initial_account_value = initial_account_value
        self.investment_per_trade = investment_per_trade
        self.fee_per_trade = fee_per_trade

    def run_strategy(self, **kwargs):
        params = kwargs
        
        open_position=False
        #while True:
            # Apply the strategy to the data
        data = self.strategy.generate_signals_backtest(self.data,**params)
        signal = self.strategy.generate_signals_trading_bot(self.data,**params)
        data = data.iloc[-1]
        
        
        ticker= self.exchange.fetch_ticker('ETH/USDT')
        price = ticker['close']
        qty = self.investment_per_trade/price
        
        print(data.Date,data.Close,data.RSI)
        if not open_position:
            if signal == 'Buy':
                print('Executing buy order',price)
                self.exchange.create_market_buy_order(symbol = self.symbol, amount = qty)
                open_position = True 
                
                
               #break

        if open_position:
           #while True:
                if signal == 'Sell':
                    print('Executing sell order',price)
                    self.exchange.create_market_sell_order(symbol = self.symbol, amount = qty)
                    open_position = False 

                    #break


def generate_parameter_combinations(param_names, param_ranges):
    param_combinations = itertools.product(*param_ranges)
    parameter_values = [{param_names[i]: p[i] for i in range(len(param_names))} for p in param_combinations]
    return parameter_values


def calculate_macd(data, **params):
    """
    Calculates the MACD line, signal line, and zero line for a given DataFrame of stock data.
    :param data: DataFrame containing stock data with at least a 'Close' column and a DatetimeIndex.
    :param ema_long_period: The period length for the long EMA. Default is 26.
    :param ema_short_period: The period length for the short EMA. Default is 12.
    :param signal_line_period: The period length for the signal line. Default is 9.
    :return: DataFrame containing the MACD line, signal line, and zero line as columns, with the same index as the input DataFrame.
    """
    ema_long = data['Close'].ewm(span=params['ema_long_period'], adjust=False).mean()
    ema_short = data['Close'].ewm(span=params['ema_short_period'], adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=params['signal_line_period'], adjust=False).mean()
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






def main():
    #RSI
    #inital_parameters = {'rsi_period': 17.0, 'buy_threshold': 21.0, 'sell_threshold': 95.0}
    #MACD
    symbol = "ETH/USDT"
    time_frame = '15m'
    df = getData(symbol,time_frame) 
    inital_parameters = {'ema_long_period': 11.0, 'ema_short_period': 6.0, 'signal_line_period': 14.0}
    strategy = MyStrategy()
    backtester = Backtester(df, strategy,100, 10, 0.0099)
    print(backtester.run_backtest(**inital_parameters))
    # rsi_data = strategy.generate_signals_backtest(df,**inital_parameters)
    # # print(rsi_data)
    #backtester.plot_backtest(**inital_parameters)
   
    # param_names = ['rsi_period','buy_threshold','sell_threshold']
    # param_ranges = [{x for x in range(1,20,1)},{x for x in range(1,30,5)},{x for x in range(60,100,5)}]
    # param_names = ['ema_long_period','ema_short_period','signal_line_period']
    # param_ranges = [{x for x in range(1,30,1)},{x for x in range(1,20,5)},{x for x in range(1,15,1)}]
    # param_combo = generate_parameter_combinations(param_names, param_ranges)
    # print(backtester.optimize_parameters(param_combo))
    # exchange = initiate_exchange()
    
    # #get_closed_order(exchange,symbol)
    # print(get_balance(exchange))
    # symbol = "ETH/USDT"
    # time_frame = '1m'
    # while True:
    #     df = getData(symbol,time_frame)  
    #     trade_bot = Trading_Bot(df,exchange,symbol,strategy,100, 10,0.0099)
    #     trade_bot.run_strategy(**inital_parameters)
    #     sleep(30)
    
    
main()





