import Utilities as util
import pandas as pd
from tqdm import tqdm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
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
        # get stop loss and take profit
        signals = util.profit_stoploss(signals,'atr')
        num_trades,win_rate,total_profit,account_value,signals = self.calculate_total_profits(signals, self.initial_account_value, self.ivestment_amount, self.fee_per_trade)
        signals.to_csv('signals.csv',index=False)
        return {'num_trades':num_trades,'win_rate': win_rate,'total_profit':total_profit,'account_value':account_value}
        
        
        
        
    def calculate_total_profits(self, data, initial_account_value, investment_amount, fee_per_trade):
        """
        Calculates the total profits from a trading strategy given a DataFrame with closing prices and trade signals.

        Args:
        - data: A Pandas DataFrame with columns "close" (closing prices), "signal" (trade signals, where 1 represents a buy signal, 0 represents a hold signal, and -1 represents a sell signal), "Stop_Loss" (stop loss percentage for each trade), and "Take_Profit" (take profit percentage for each trade).
        - initial_account_value: The total amount of money in the account at the start of the trading period.
        - investment_amount: The dollar amount spent on each trade.
        - fee_per_trade: The fee associated with each trade.

        Returns:
        - The total profits (or losses) from the trading strategy.
        """
        account_value = initial_account_value
        num_shares = 0
        num_trades = 0
        total_profit = 0
        
        data['Account Value'] = initial_account_value
        data['Gains/Losses'] = 0
        purchase_price = 0
        stop_loss_price = None
        take_profit_price =  None 
        take_profit_value = None
        stop_loss_value =  None
        position = None
        
        for i in range(len(data)):
            if data['Signal'][i] == 1: # Buy signal
                # Calculate how many tokens were purchased with the investment amount, factoring in the trading fee
                num_shares = (investment_amount - (fee_per_trade*100)) / data['Close'][i]
                # Deduct the total investment amount and trading fee from the account value
                #account_value -= investment_amount 
                data['Account Value'][i] = account_value
                purchase_price = data['Close'][i]
                stop_loss_price = data['Stop_Loss'][i]
                take_profit_price =  data['Take_Profit'][i] 
               
                take_profit_value = num_shares * (take_profit_price - purchase_price)
                stop_loss_value =  num_shares * (stop_loss_price - purchase_price)
                position = 'Long'


            # Check if price reaches take profit or stop loss
            if num_shares > 0 and position == 'Long':
                if data['Close'][i] >= take_profit_price:
                    # Sell all shares and calculate profit
                    account_value += take_profit_value
                    data['Account Value'][i] = account_value
                    gains_losses = (take_profit_price - purchase_price) * num_shares
                    data['Gains/Losses'][i] = gains_losses
                   
                    total_profit += gains_losses
                    num_shares = 0
                elif data['Close'][i] <= stop_loss_price:
                    # Sell all shares and calculate loss
                    account_value += stop_loss_value
                    data['Account Value'][i] = account_value
                    gains_losses = (stop_loss_price - purchase_price) * num_shares
                    data['Gains/Losses'][i] = gains_losses
                    total_profit += gains_losses
                    num_shares = 0

            elif data['Signal'][i] == -1: # Buy signal
                # Calculate how many tokens were purchased with the investment amount, factoring in the trading fee
                num_shares = (investment_amount - (fee_per_trade*100)) / data['Close'][i]
                # Deduct the total investment amount and trading fee from the account value
                #account_value -= investment_amount 
                data['Account Value'][i] = account_value
                purchase_price = data['Close'][i]
                stop_loss_price = data['Stop_Loss'][i]
                take_profit_price =  data['Take_Profit'][i] 
            
                take_profit_value = num_shares * (take_profit_price - purchase_price)
                stop_loss_value =  num_shares * (stop_loss_price - purchase_price)
                position = 'Short'

            # Check if price reaches take profit or stop loss
            if num_shares > 0 and position == 'Short':
                if data['Close'][i] >= stop_loss_price:
                    # Sell all shares and calculate profit
                    account_value += take_profit_value
                    data['Account Value'][i] = account_value
                    gains_losses = (take_profit_price - purchase_price) * num_shares
                    data['Gains/Losses'][i] = gains_losses
                   
                    total_profit += gains_losses
                    num_shares = 0
                elif data['Close'][i] <= take_profit_price:
                    # Sell all shares and calculate loss
                    account_value += stop_loss_value
                    data['Account Value'][i] = account_value
                    gains_losses = (stop_loss_price - purchase_price) * num_shares
                    data['Gains/Losses'][i] = gains_losses
                    total_profit += gains_losses
                    num_shares = 0
            elif data['Signal'][i] == 0: # Hold signal
                 data['Account Value'][i] = account_value

        num_positive_gains_losses = (data['Gains/Losses'] > 0).sum()
        num_trades =  len(data.loc[(data["Signal"] == 1) | (data["Signal"] == -1)])
        win_rate = num_positive_gains_losses/num_trades
            
        return num_trades,win_rate,total_profit,account_value,data
    
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
            total_profit_ratio = results_df['total_profit'].max()
            total_profit_params = results_df.loc[results_df['total_profit'].idxmax()].to_dict()
        except:
            total_profit_ratio = 0
            total_profit_params = {}
        
        return {'total_profit_ratio': total_profit_ratio, 'total_profit_ratio_params': total_profit_params}
                
                
    
    # def calculate_sharpe_ratio(self,data):
    #     """
    #     Calculate the Sharpe Ratio from a DataFrame of daily closing prices.

    #     Parameters:
    #     df_prices (pandas.DataFrame): A DataFrame of daily closing prices.

    #     Returns:
    #     float: The Sharpe Ratio.
    #     """

    #     # Calculate the daily returns based on the trade signals
    #     data['Returns'] = data['Signal'] * (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)

    #     # Calculate the Sharpe ratio
    #     R_p = data['Returns'].mean() * 252
    #     R_f = 0.02 # Assume a risk-free rate of 2%
    #     sigma_p = data['Returns'].std() * np.sqrt(252)
    #     sharpe_ratio = (R_p - R_f) / sigma_p


    #     return sharpe_ratio
    

    
    def plot_backtest(self,indicator=None, **params):    
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()
        
        
        # Apply the strategy to the data
        signals = self.strategy.generate_signals_backtest(data,self.selected_strategy,**params)
        # get stop loss and take profit
        signals = util.profit_stoploss(signals,'atr')
        # Create a candlestick chart of the data
        candlestick = go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )

        # Create a scatter plot of the buy and
        # sell signals
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
        # Add stop loss and take profit lines to the chart
        stop_loss_lines = []
        take_profit_lines = []

        for index, row in signals.iterrows():
            if row['Signal'] == 1:
                stop_loss_lines.append(
                    go.layout.Shape(
                        type="line",
                        x0=row['Date'],
                        y0=row['Stop_Loss'],
                        x1=row['Date'] + pd.Timedelta(hours=23, minutes=59, seconds=59),
                        y1=row['Stop_Loss'],
                        line=dict(
                            color="red",
                            width=2,
                            dash="dash"
                        )
                    )
                )
                take_profit_lines.append(
                    go.layout.Shape(
                        type="line",
                        x0=row['Date'],
                        y0=row['Take_Profit'],
                        x1=row['Date'] + pd.Timedelta(hours=23, minutes=59, seconds=59),
                        y1=row['Take_Profit'],
                        line=dict(
                            color="green",
                            width=2,
                            dash="dash"
                        )
                    )
                )
            elif row['Signal'] == -1:
                stop_loss_lines.append(
                    go.layout.Shape(
                        type="line",
                        x0=row['Date'],
                        y0=row['Stop_Loss'],
                        x1=row['Date'] + pd.Timedelta(hours=23, minutes=59, seconds=59),
                        y1=row['Stop_Loss'],
                        line=dict(
                            color="green",
                            width=2,
                            dash="dash"
                        )
                    )
                )
                take_profit_lines.append(
                    go.layout.Shape(
                        type="line",
                        x0=row['Date'],
                        y0=row['Take_Profit'],
                        x1=row['Date'] + pd.Timedelta(hours=23, minutes=59, seconds=59),
                        y1=row['Take_Profit'],
                        line=dict(
                            color="red",
                            width=2,
                            dash="dash"
                        )
                    )
                )

        # Create a layout for the chart with a secondary axis for the RSI plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # Add the candlestick chart to the first row of the subplot
        # Add the candlestick chart and buy and sell signals to the first row
        fig.add_trace(candlestick, row=1, col=1)
        fig.add_trace(buy_scatter, row=1, col=1)
        fig.add_trace(sell_scatter, row=1, col=1)

        # Add the stop loss and take profit lines to the first row
        for stop_loss_line in stop_loss_lines:
            fig.add_shape(stop_loss_line, row=1, col=1)

        for take_profit_line in take_profit_lines:
            fig.add_shape(take_profit_line, row=1, col=1)

        # Add the RSI plot to the second row
        if indicator == 'rsi':
            rsi = self.indicators.rsi(data, self.selected_strategy['rsi_window'])
            fig.add_trace(
                go.Scatter(
                    x=data['Date'],
                    y=rsi,
                    name='RSI',
                    line=dict(color='black')
                ),
                row=2,
                col=1
            )

            # Add RSI overbought and oversold regions to the chart
            fig.add_shape(
                go.layout.Shape(
                    type="rect",
                    x0=data['Date'].iloc[0],
                    y0=70,
                    x1=data['Date'].iloc[-1],
                    y1=100,
                    fillcolor="lightgrey",
                    opacity=0.5,
                    line=dict(color='lightgrey', width=0),
                ),
                row=2,
                col=1
            )

            fig.add_shape(
                go.layout.Shape(
                    type="rect",
                    x0=data['Date'].iloc[0],
                    y0=0,
                    x1=data['Date'].iloc[-1],
                    y1=30,
                    fillcolor="lightgrey",
                    opacity=0.5,
                    line=dict(color='lightgrey', width=0),
                ),
                row=2,
                col=1
            )

            # Update the layout
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                title=f"{self.selected_strategy['name']} Backtest Results",
                yaxis_title="Price",
                yaxis2_title="RSI" if indicator == 'rsi' else "",
                height=800
            )

        # Show the chart
        fig.show()
       
       
    