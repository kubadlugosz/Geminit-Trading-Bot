import Utilities as util
import pandas as pd

class MyStrategy:
    
    def generate_signals_backtest(self,data,user_input,**params):
       
        # data['Signal'] = data['Signal'].astype("float")
        if user_input == 'MACD':
            signals = self.MACD_strategy(data,**params)
        elif user_input == 'RSI':
            signals = self.RSI_strategy(data,**params)
        elif user_input == 'Stochastic':
            signals = self.Stochastic_strategy(data,**params)
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
        macd_df = util.calculate_macd(data, **params)
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
    
    def Stochastic_strategy(self,data,**params):
        #stochastic ocillator trading strategy
        stoch_df = util.stochastic_oscillator(data, **params)
        stoch_df = util.crossover(stoch_df,stoch_df['K'],stoch_df['D'])
        data = stoch_df.dropna()
        data = data.reset_index(drop=True)
        
        # Calculate the 200-day EMA on the Close price
        data['EMA'] = data['Close'].ewm(span=200).mean()
        # Set a flag to track if we're currently in a position
        in_position = False
        
        # Loop through each row and set the signal based on the RSI and previous position
        data['Signal'] = 0

        for i in range(1, len(data)):

            ema = data['EMA'][i]
            crossover = data['Crossover'][i]
            price = data['Close'][i]
            #condition for buy signal
            if crossover == 1 and price > ema:
                data['Signal'][i] = 1
            #condition for sell signal
            elif crossover == -1 and price < ema:
                data['Signal'][i] = -1
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