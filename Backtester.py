import Utilities as util
import pandas as pd
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
        # total_profits = self.calculate_total_profits(signals, self.initial_account_value, self.ivestment_amount, self.fee_per_trade)
        # print(total_profits)
        signals.to_csv('signals.csv',index=False)
        
        
        
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
                account_value -= investment_amount 
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
                account_value -= investment_amount 
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

                
            
        return total_profit
                
                
                
            # elif data['Signal'][i] == -1: # Sell signal
            #     # Calculate the gross proceeds from selling the tokens, factoring in the trading fee
            #     gross_proceeds = num_shares * data['Close'][i] - (fee_per_trade*100)
            #     # Calculate the profit or loss from the trade
            #     profit_loss = gross_proceeds - investment_amount
            #     # Add the profit or loss to the total profit
            #     total_profit += profit_loss
            #     # Add the gross proceeds (minus trading fee) to the account value
            #     account_value += gross_proceeds
            #     # Reset the number of shares to 0
            #     num_shares = 0
            #     # Increment the number of trades counter
            #     num_trades += 1

            #     data['Account Value'][i] = account_value
            #     data['Gains/Losses'][i] = profit_loss
            # else: # Hold signal
            #     data['Account Value'][i] = account_value

        # Add the final account value to the total profit
        # total_profit += account_value - initial_account_value


        # account_value, total_profits, num_trades = self.calculate_profit(signals, self.initial_account_value, self.ivestment_amount, self.fee_per_trade)
        # print(account_value, total_profits, num_trades )
        # signals.to_csv('signals.csv',index=False)


    def calculate_profit(self,data, initial_account_value, investment_amount, fee_per_trade):
    # Initialize account value, number of trades, and profits to zero
        account_value = initial_account_value
        num_trades = 0
        total_profits = 0
    
        # Loop through each row in the dataframe
        for i, row in data.iterrows():
            # Check if there is a trade signal for this row
            if row['Signal'] != 0:
                # Calculate the number of shares to buy/sell
                shares = investment_amount / row['Close']
                
                # Calculate the transaction cost
                transaction_cost = shares * row['Close'] * fee_per_trade
                
                # Subtract the transaction cost from the account value
                account_value -= transaction_cost
                
                # Calculate the profit/loss for this trade
                if row['Signal'] == 1:
                    # Buy trade
                    profit = shares * (row['Take_Profit'] - row['Close'])
                else:
                    # Sell trade
                    profit = shares * (row['Close'] - row['Stop_Loss'])
                
                # Add the profit to the total profits
                total_profits += profit
                
                # Add the profit to the account value
                account_value += profit
                
                # Increment the number of trades
                num_trades += 1
        
        # Return the account value, total profits, and number of trades
        return account_value, total_profits, num_trades


        def win_rate(signals):
            # Count the number of winning trades and losing trades
            num_winning_trades = (signals['Signal'] == 1) & (signals['Take_Profit'] > signals['Close'])
            num_losing_trades = (signals['Signal'] == -1) & (signals['Stop_Loss'] < signals['Close'])
            
            # Calculate the win rate
            win_rate = num_winning_trades.sum() / (num_winning_trades.sum() + num_losing_trades.sum())
            
            return win_rate






















    # def run_backtest(self, **kwargs):
    #     params = kwargs
    #     # Create a copy of the data to avoid modifying the original
    #     data = self.data.copy()
        
    #     # Apply the strategy to the data
    #     signals = self.strategy.generate_signals_backtest(data,self.selected_strategy,**params)
    #     # Calculate the total profit in whole dollars
    #     total_profit,account_value,signals= self.calculate_total_profits(signals,self.initial_account_value,self.ivestment_amount,self.fee_per_trade)
        
    #     # Calculate sharpe ratio
    #     sharpe_ratio = self.calculate_sharpe_ratio(signals)
       
    #     # Calculate win rate
    #     win_rate = self.calculate_win_rate(signals)
    #     signals.to_csv('signals.csv')
    #     # Return the results
    #     return {'sharpe_ratio': sharpe_ratio, 'innitial_account_value':self.initial_account_value, 'account_value': account_value,'total_profit': total_profit, 'win_rate': win_rate}
      
        

    # def calculate_total_profits(self, data, initial_account_value, investment_amount, fee_per_trade):
    #     """
    #     Calculates the total profits from a trading strategy given a DataFrame with closing prices and trade signals.

    #     Args:
    #     - data: A Pandas DataFrame with columns "close" (closing prices) and "signal" (trade signals, where 1 represents a buy signal, 0 represents a hold signal, and -1 represents a sell signal).
    #     - initial_account_value: The total amount of money in the account at the start of the trading period.
    #     - dollar_amount_spent_per_trade: The dollar amount spent on each trade.
    #     - fee_per_trade: The fee associated with each trade.

    #     Returns:
    #     - The total profits (or losses) from the trading strategy.
    #     """
    #     account_value = initial_account_value
    #     num_shares = 0
    #     num_trades = 0
    #     total_profit = 0
    #     data['Account Value'] = 0
    #     data['Gains/Losses'] = 0
    #     for i in range(len(data)):
    #         if data['Signal'][i] == 1: # Buy signal
    #             # Calculate how many tokens were purchased with the investment amount, factoring in the trading fee
    #             num_shares = (investment_amount - (fee_per_trade*100)) / data['Close'][i]
    #             # Deduct the total investment amount and trading fee from the account value
    #             account_value -= investment_amount 
    #             data['Account Value'][i] = account_value
    #         elif data['Signal'][i] == -1: # Sell signal
    #             # Calculate the gross proceeds from selling the tokens, factoring in the trading fee
    #             gross_proceeds = num_shares * data['Close'][i] - (fee_per_trade*100)
    #             # Calculate the profit or loss from the trade
    #             profit_loss = gross_proceeds - investment_amount
    #             # Add the profit or loss to the total profit
    #             total_profit += profit_loss
    #             # Add the gross proceeds (minus trading fee) to the account value
    #             account_value += gross_proceeds
    #             # Reset the number of shares to 0
    #             num_shares = 0
    #             # Increment the number of trades counter
    #             num_trades += 1

    #             data['Account Value'][i] = account_value
    #             data['Gains/Losses'][i] = profit_loss
    #         else: # Hold signal
    #             data['Account Value'][i] = account_value

    #     # Add the final account value to the total profit
    #     total_profit += account_value - initial_account_value

        
        
    #     return total_profit,account_value,data
    
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
    
    # def calculate_win_rate(self,data):
    #     """
    #     Calculate the win rate for a trading strategy based on a DataFrame of closing prices and signals.

    #     Parameters:
    #     data (pandas.DataFrame): A DataFrame containing closing prices and signals.

    #     Returns:
    #     float: The win rate as a percentage.
    #     """
    #     # Calculate the number of winning and losing trades
    #     num_wins = len(data[data['Gains/Losses'] > 0])
    #     num_losses = len(data[data['Gains/Losses'] < 0])
        
    #     if num_losses == 0 and num_wins == 0:
    #         win_rate = 0
    #     # Calculate the win rate as a percentage
    #     else:
    #         win_rate = num_wins / (num_wins + num_losses) * 100

    #     return win_rate
    
    # def plot_backtest(self,indicator=None, **params):    
    #     # Create a copy of the data to avoid modifying the original
    #     data = self.data.copy()
        
        
    #     # Apply the strategy to the data
    #     signals = self.strategy.generate_signals_backtest(data,self.selected_strategy,**params)
    #     # Create a candlestick chart of the data
    #     candlestick = go.Candlestick(
    #         x=data['Date'],
    #         open=data['Open'],
    #         high=data['High'],
    #         low=data['Low'],
    #         close=data['Close']
    #     )

    #     # Create a scatter plot of the buy and sell signals
    #     buys = signals[signals['Signal'] == 1]
    #     sells = signals[signals['Signal'] == -1]

    #     buy_scatter = go.Scatter(
    #         x=buys['Date'],
    #         y=buys['Close'],
    #         mode='markers',
    #         name='Buy',
    #         marker=dict(
    #             symbol='triangle-up',
    #             size=10,
    #             color='green'
    #         )
    #     )

    #     sell_scatter = go.Scatter(
    #         x=sells['Date'],
    #         y=sells['Close'],
    #         mode='markers',
    #         name='Sell',
    #         marker=dict(
    #             symbol='triangle-down',
    #             size=10,
    #             color='red'
    #         )
    #     )
        
    #     # Create a layout for the chart with a secondary axis for the RSI plot
    #     fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    #     # Add the candlestick chart to the first row of the subplot
    #     fig.add_trace(candlestick, row=1, col=1)

    #     # Add the buy and sell signals to the first row of the subplot
    #     fig.add_trace(buy_scatter, row=1, col=1)
    #     fig.add_trace(sell_scatter, row=1, col=1)

    #     # Create a plot of the RSI indicator, if provided, and add it to the second row of the subplot
    #     if indicator is not None:
    #         rsi_trace = go.Scatter(
    #             x=data['Date'],
    #             y=indicator,
    #             name='RSI'
    #         )
    #         fig.add_trace(rsi_trace, row=2, col=1)

    #     # Update the layout to include a title and axis labels
    #     fig.update_layout(
    #         title='Trading Signals',
    #         xaxis=dict(title='Date', tickformat='%Y-%m-%d %H:%M:%S'),
    #         yaxis=dict(title='Price', domain=[0.2, 1]),
    #         yaxis2=dict(title='RSI', domain=[0, 0.15])
    #     )

    #     # Display the figure
    #     fig.show()
       
       
    
    # def optimize_parameters(self, parameter_values):
    #     results = []
    #     with tqdm(total=len(parameter_values)) as pbar:
    #         for params in parameter_values:
    #             try:
    #                 result = self.run_backtest(**params)
    #                 result.update(params)
    #                 results.append(result)
    #                 pbar.update(1)
    #             except:
    #                 continue
            
    #     results_df = pd.DataFrame(results)
    #     try:
    #         max_win_ratio = results_df['win_rate'].max()
    #         max_win_ratio_params = results_df.loc[results_df['win_rate'].idxmax()].to_dict()
    #     except:
    #         max_win_ratio = 0
    #         max_win_ratio_params = {}
        
    #     return {'max_sharpe_ratio': max_win_ratio, 'max_sharpe_ratio_params': max_win_ratio_params}