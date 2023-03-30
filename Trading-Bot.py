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

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
from binance.client import Client
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")


def getData(symbol,time,client):
    """
    1504541580000, // UTC timestamp in milliseconds, integer
    4235.4,        // (O)pen price, float
    4240.6,        // (H)ighest price, float
    4230.0,        // (L)owest price, float
    4230.7,        // (C)losing price, float
    37.72941911    // (V)olume float (usually in terms of the base currency, the exchanges docstring may list whether quote or base units are used)
    """
    klines = client.get_historical_klines(symbol=symbol, interval=time)

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(klines, columns=columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/Chicago')

    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    return df

class Backtester:
    def __init__(self, data, strategy,selected_strategy,initial_account_value, ivestment_amount,fee_per_trade):
        self.data = data
        self.strategy = strategy
        self.selected_strategy = selected_strategy
        self.initial_account_value = initial_account_value
        self.ivestment_amount = ivestment_amount
        self.fee_per_trade = fee_per_trade


    def run_backtest(self, **kwargs):
        params = kwargs
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        # Apply the strategy to the data
        signals = self.strategy.generate_signals_backtest(data,self.selected_strategy,**params)
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
        
        if num_losses == 0 and num_wins == 0:
            win_rate = 0
        # Calculate the win rate as a percentage
        else:
            win_rate = num_wins / (num_wins + num_losses) * 100

        return win_rate
    
    def plot_backtest(self,indicator=None, **params):    
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        
        # Apply the strategy to the data
        signals = self.strategy.generate_signals_backtest(data,self.selected_strategy,**params)
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
        try:
            max_win_ratio = results_df['win_rate'].max()
            max_win_ratio_params = results_df.loc[results_df['win_rate'].idxmax()].to_dict()
        except:
            max_win_ratio = 0
            max_win_ratio_params = {}
        
        return {'max_sharpe_ratio': max_win_ratio, 'max_sharpe_ratio_params': max_win_ratio_params}
    
    
    
    
    

class MyStrategy:
    
    def generate_signals_backtest(self,data,user_input,**params):
       
        # data['Signal'] = data['Signal'].astype("float")
        if user_input == 'MACD':
            signals = self.MACD_strategy(data,**params)
        elif user_input == 'RSI':
            signals = self.RSI_strategy(data,**params)
        return signals 
    
    def RSI_strategy(self,data,**params):
        from ta.momentum import rsi
        
        # Define your trading signals here
        lower_limit= params['Buy Threshold'] 
        upper_limit = params['Sell Threshold']
        
        # Calculate the RSI indicator
        #data["RSI"] = rsi(data["Close"],window=params['RSI Period'])
        data['RSI']= ta.momentum.RSIIndicator(data["Close"], window=params['RSI Period']).rsi()
       
        #data = rsi_calculation(data,params['RSI Period'])

        #data["RSI"] = ta.momentum.RSIIndicator(close=data["close"], window=params['RSI Period']).rsi()
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
        return data
            
        
        
    def MACD_strategy(self,data,**params):
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
    
    def generate_signals_trading_bot(self,data,user_input,**params):
        if user_input == 'RSI':
            rsi = data['RSI'].iloc[-1]
            if rsi <= params['Buy Threshold'] :
                return 'Buy'
            elif rsi >= params['Sell Threshold'] :
                return 'Sell'
            else:
                return 'Neutral'
        elif user_input == 'MACD':
            macd = data['MACD_line']
            zero_line = data['Zero_line']
            signal_line = data['Signal_line']
            if (macd < signal_line) & (macd > zero_line) & (signal_line > zero_line): 
                return 'Sell'
            elif (macd > signal_line) & (macd < zero_line) & (signal_line < zero_line):
                return 'Buy'
            else:
                 return 'Neutral'
            
        
        
 

class Trading_Bot:
    def __init__(self,user_input,exchange,symbol,time_frame,strategy,investment_per_trade):
    
        self.user_input = user_input
        self.exchange = exchange
        self.symbol = symbol
        self.time_frame = time_frame
        self.strategy = strategy
        self.investment_per_trade = investment_per_trade
      

    def run_strategy(self, **kwargs):
        params = kwargs
        
        open_position=False
       
        while True:
            df = getData(self.symbol,self.time_frame,self.exchange)
            # Apply the strategy to the data
            data = self.strategy.generate_signals_backtest(df,self.user_input,**params)
            print(data.tail(1).to_string(index=False))
            data = data.iloc[-1]
            signal = self.strategy.generate_signals_trading_bot(data,self.user_input,**params)
            
            
            
            ticker= self.exchange.get_symbol_ticker(symbol=self.symbol)
            price = float(ticker['price'])
            
            qty = self.investment_per_trade/price
            qty = round(float(qty),4)
            print(signal)
          
            sleep(60)
            
            if not open_position:
                if signal == 'Buy':
                    print('Executing buy order',price)
                    
                                     
                    self.exchange.create_order(symbol=self.symbol,side='BUY',type='MARKET',quantity=str(qty))
                   
                    open_position = True 
                    account = self.exchange.get_account()['balances'][6]['free']
                    create_log(data.Date,price,'BUY',qty,account,0)
                    
                    break

        if open_position:
            while True:
                df = getData(self.symbol,self.time_frame,self.exchange)
                # Apply the strategy to the data
                data = self.strategy.generate_signals_backtest(df,self.user_input,**params)
                print(data.tail(1).to_string(index=False))
                data = data.iloc[-1]
                signal = self.strategy.generate_signals_trading_bot(data,self.user_input,**params)
                
                
                
                ticker= self.exchange.get_symbol_ticker(symbol=self.symbol)
                price = float(ticker['price'])
                
                qty = self.investment_per_trade/price
                qty = round(float(qty),4)
                
                print(signal)
                if signal == 'Sell':
                    print('Executing sell order',price)
                    self.exchange.create_order(symbol=self.symbol,side='SELL',type='MARKET',quantity=str(qty))
                    open_position = False 
                    account = self.exchange.get_account()['balances'][6]['free']
                    create_log(data.Date,price,'SELL',qty,account,0)
                  
                    break
                sleep(60)



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

class TradingApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.title("Trading App")

        # Set the window size
        self.geometry("500x500")

        # Create the symbol label and dropdown
        symbol_label = ttk.Label(self, text="Symbol:")
        symbol_label.pack(side=tk.TOP, padx=10, pady=10)
        self.symbol_var = tk.StringVar()
        self.symbol_dropdown = ttk.Combobox(self, textvariable=self.symbol_var,
                                            values=['BTCUSDT', 'ETHUSDT', 'LTCUSDT'])
        self.symbol_dropdown.pack(side=tk.TOP, padx=10, pady=5)

        # Create the time frame label and dropdown
        time_frame_label = ttk.Label(self, text="Time Frame:")
        time_frame_label.pack(side=tk.TOP, padx=10, pady=10)
        self.time_frame_var = tk.StringVar()
        self.time_frame_dropdown = ttk.Combobox(self, textvariable=self.time_frame_var,
                                                values=['1m', '5m', '15m', '1h', '4h', '1d'])
        self.time_frame_dropdown.pack(side=tk.TOP, padx=10, pady=5)

        # Create the initial investment amount label and input field
        account_label = ttk.Label(self, text="Initial Investment Amount:")
        account_label.pack(side=tk.TOP, padx=10, pady=10)
        self.account_entry = ttk.Entry(self)
        self.account_entry.pack(side=tk.TOP, padx=10, pady=5)

        # Create the investment amount label and input field
        investment_label = ttk.Label(self, text="Investment Amount:")
        investment_label.pack(side=tk.TOP, padx=10, pady=10)
        self.investment_entry = ttk.Entry(self)
        self.investment_entry.pack(side=tk.TOP, padx=10, pady=5)

        # Create the fee per trade label and input field
        fee_label = ttk.Label(self, text="Fee Per Trade:")
        fee_label.pack(side=tk.TOP, padx=10, pady=10)
        self.fee_entry = ttk.Entry(self)
        self.fee_entry.pack(side=tk.TOP, padx=10, pady=5)

        # Create the mode label and dropdown
        mode_label = ttk.Label(self, text="Choose Mode:")
        mode_label.pack(side=tk.TOP, padx=10, pady=10)
        self.mode_var = tk.StringVar()
        self.mode_dropdown = ttk.Combobox(self, textvariable=self.mode_var,
                                          values=['Backtest', 'Plot Backtest', 'Optimize', 'Trade'])
        self.mode_dropdown.pack(side=tk.TOP, padx=10, pady=5)

        # Create the strategy label and dropdown
        strategy_label = ttk.Label(self, text="Choose Strategy:")
        strategy_label.pack(side=tk.TOP, padx=10, pady=10)
        self.strategy_var = tk.StringVar()
        self.strategy_dropdown = ttk.Combobox(self, textvariable=self.strategy_var,
                                              values=['RSI', 'MACD'])
        self.strategy_dropdown.pack(side=tk.TOP, padx=10, pady=5)

        # Create the button to start trading
        start_button = ttk.Button(self, text="Start Trading", command=self.start_trading)
        start_button.pack(side=tk.TOP, padx=10, pady=10)

    def start_trading(self):
        # Get the selected symbol, time frame, investment amount, and fee per trade
        symbol = self.symbol_var.get()
        time_frame = self.time_frame_var.get()
        account_amount = float(self.account_entry.get())
        investment_amount = float(self.investment_entry.get())
        fee_per_trade = float(self.fee_entry.get())

        # Get the selected mode and strategy
        mode = self.mode_var.get()
        selected_strategy = self.strategy_var.get()
        exchange = Client(api_key=config.key,api_secret=config.secret,tld='us',testnet=True)
        df = getData(symbol,time_frame,exchange)
        strategy = MyStrategy()
        backtester = Backtester(df, strategy,selected_strategy,account_amount, investment_amount, fee_per_trade)   
        param_values = {}
        if mode == "Backtest":
            # Get the parameter names for the selected strategy
            if selected_strategy == "RSI":
                param_names = ["RSI Period", "Buy Threshold", "Sell Threshold"]
            elif selected_strategy == "MACD":
                param_names = ["EMA Long Period", "EMA Short Period", "Signal Line Period"]
            else:
                # Invalid strategy selected
                messagebox.showerror("Error", "Please select a valid strategy.")
                return

            # Create a dialog box to get the parameter values from the user
            
            for param_name in param_names:
                param_value = simpledialog.askfloat(param_name, f"Please enter the {param_name}")
                if param_value is None:
                    # User canceled the input dialog box
                    return
                #param_values.append(param_value)
                param_values[param_name] = param_value
            print(param_values)
            # Run the backtest with the selected strategy and parameters
             
            print(backtester.run_backtest(**param_values))

        elif mode == "Plot Backtest":

            backtester.plot_backtest(**param_values)
        
        elif mode == "Optimize":
            if selected_strategy == "RSI":
                param_names = ["RSI Period", "Buy Threshold", "Sell Threshold"]
            elif selected_strategy == "MACD":
                param_names = ["EMA Long Period", "EMA Short Period", "Signal Line Period"]
            else:
                # Invalid strategy selected
                messagebox.showerror("Error", "Please select a valid strategy.")

            # Create a separate window to display the parameter names and input fields for start, stop, and step
            optimize_window = tk.Toplevel(self)
            optimize_window.title("Optimize Parameters")

            # Create a label for each parameter name and input fields for start, stop, and step
            param_entries = []
            for param_name in param_names:
                label = ttk.Label(optimize_window, text=param_name)
                label.pack(side=tk.TOP, padx=10, pady=5)

                start_entry = ttk.Entry(optimize_window)
                start_entry.pack(side=tk.TOP, padx=10, pady=5)
                stop_entry = ttk.Entry(optimize_window)
                stop_entry.pack(side=tk.TOP, padx=10, pady=5)
                step_entry = ttk.Entry(optimize_window)
                step_entry.pack(side=tk.TOP, padx=10, pady=5)

                param_entries.append((param_name, start_entry, stop_entry, step_entry))

            # Create a button to run the optimization with the specified parameters
            def run_optimization():
                param_values = []
                param_names=[]
                for param_entry in param_entries:
                    param_name = param_entry[0]
                    start = float(param_entry[1].get())
                    stop = float(param_entry[2].get())
                    step = float(param_entry[3].get())
                    values = {x for x in range(int(start), int(stop), int(step))}
                    param_values.append(values)
                    param_names.append(param_name)
              
                param_combo = generate_parameter_combinations(param_names, param_values)
                
                print(backtester.optimize_parameters(param_combo))
                

            # Create a button to run the optimization
            optimize_button = ttk.Button(optimize_window, text="Optimize", command=run_optimization)
            optimize_button.pack(side=tk.TOP, padx=10, pady=10)

            # Show the optimize window
            optimize_window.mainloop()
        elif mode == 'Trade':
            
            #TODO impletment trading functionality
            pass 
            

                        









def main():
    
    exchange = Client(api_key=config.key,api_secret=config.secret,tld='us',testnet=True)
    # #MACD
    symbol = "ETHUSDT"
    time_frame = '15m'
    df = getData(symbol,time_frame,exchange) 
    strategy = MyStrategy()
    backtester = Backtester(df, strategy,'RSI',1000, 500, 0.0099)
 
        
    #exchange = initiate_exchange()   
    
    # inital_parameters = {'EMA Long Period': 11.0, 'EMA Short Period': 6.0, 'Signal Line Period': 14.0}
    # trade_bot = Trading_Bot('MACD',exchange,symbol,time_frame,strategy,100)
    
    # trade_bot.run_strategy(**inital_parameters)
       
    # print(backtester.run_backtest(**inital_parameters))
    # backtester.plot_backtest(**inital_parameters)
    #Run optimization
    #param_name = ['EMA Long Period','EMA Short Period', 'Signal Line Period']
    param_name = ["RSI Period", "Buy Threshold", "Sell Threshold"]
    param_value = [{x for x in range(1, 30, 1)},{x for x in range(1, 30, 1)},{x for x in range(70, 100, 1)}]
    param_combo = generate_parameter_combinations(param_name, param_value)
    backtester.optimize_parameters(**param_combo)

    # app = TradingApp()
    
    # # Start the Tk mainloop
    # app.mainloop()

   
main()





