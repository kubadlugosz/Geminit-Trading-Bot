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
            # get stop loss and take profit
            data = util.profit_stoploss(data,'atr')
            print(data.tail(1).to_string(index=False))
            data = data.iloc[-1]
            #signal = self.strategy.generate_signals_trading_bot(data,self.user_input,**params)
            signal = data['Signal']
            take_profit = data['Take_profit']
            stop_loss = data['Stop_loss']
            ticker= self.exchange.get_symbol_ticker(symbol=self.symbol)
            price = float(ticker['price'])
            
            qty = self.investment_per_trade/price
            qty = round(float(qty),4)
            print(signal)
          
            sleep(60)
            
            
            if signal == 1:
                    print('Executing buy order',price)
                    
                                     
                    self.exchange.client.order_market_buy(
                                                        symbol=self.symbol,
                                                        quantity=qty
                                                    )  # initial opening order
                    self.exchange.client.create_oco_order(
                                                        symbol=self.symbol,
                                                        side=self.exchange.client.SIDE_SELL,
                                                        quantity=qty,
                                                        stopPrice=stop_loss,
                                                        stopLimitPrice=stop_loss,
                                                        price=stop_loss,
                                                        stopLimitTimeInForce='GTC',
                                                    )
                    self.exchange.client.create_oco_order(
                                                        symbol=self.symbol,
                                                        side=self.exchange.client.SIDE_SELL,
                                                        quantity=qty,
                                                        stopPrice=take_profit,
                                                        stopLimitPrice=take_profit,
                                                        price=take_profit,
                                                        stopLimitTimeInForce='GTC',
                                                    )
                                                                    
                  
                    account = self.exchange.get_account()['balances'][6]['free']
                    util.create_log(data.Date,price,'BUY',qty,account,0)
                    
                    break

       