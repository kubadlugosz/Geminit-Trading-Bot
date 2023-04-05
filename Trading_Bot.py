from timer import sleep 
import Utilities as util
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
            df = util.getData(self.symbol,self.time_frame,self.exchange)
            # Apply the strategy to the data
            data = self.strategy.generate_signals_backtest(df,self.user_input,**params)
            print(data.tail(1).to_string(index=False))
            data = data.iloc[-1]
            #signal = self.strategy.generate_signals_trading_bot(data,self.user_input,**params)
            
            
            
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
                    util.create_log(data.Date,price,'BUY',qty,account,0)
                    
                    break

        if open_position:
            while True:
                df = util.getData(self.symbol,self.time_frame,self.exchange)
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
                    util.create_log(data.Date,price,'SELL',qty,account,0)
                  
                    break
                sleep(60)