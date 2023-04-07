import time
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
        
        
        open_position = False
        order_ids = {}
        while True:
            df = util.getData(self.symbol,self.time_frame)
            # Apply the strategy to the data
            data = self.strategy.generate_signals_backtest(df,self.user_input,**params)
            # get stop loss and take profit
            data = util.profit_stoploss(data,'atr')
            print(data.tail(1).to_string(index=False))
            data = data.iloc[-1]
           
            #signal = self.strategy.generate_signals_trading_bot(data,self.user_input,**params)
            signal = data['Signal']
            take_profit = data['Take_Profit']
            stop_loss = data['Stop_Loss']
            ticker= self.exchange.fetch_ticker(symbol=self.symbol)
            price = float(ticker['last'])
            
            qty = self.investment_per_trade/price
            qty = round(float(qty),4)
            print(signal)
          
           
            
            time.sleep(int(self.time_frame.replace('m', ''))*60)
            if signal == 1 and not open_position:
                    # we will be excuting a long position
                    print('Executing long order',price,"with take profit",take_profit,"and stop loss",stop_loss)
                    
                                     
                    self.exchange.create_order(
                        symbol=self.symbol,
                        type='market',
                        side='buy',
                        amount=qty
                    )

                    self.exchange.create_order(
                        symbol=self.symbol,
                        type='stop',
                        side='sell',
                        amount=qty,
                        price=stop_loss,
                        params={
                            'stopPrice': stop_loss
                        }
                    )

                    self.exchange.create_order(
                        symbol=self.symbol,
                        type='take_profit_market',
                        side='sell',
                        amount=qty,
                        price=take_profit,
                        params={
                            'closePosition': True,
                            'stopPrice': take_profit
                        }
                    )
                                                                    
                  
                    account = self.exchange.fetch_balance()['USDT']['Total']
                    util.create_log(data.Date,price,'BUY',qty,account,0)
                    
                    break
            
            elif signal == -1:
                 # we will be excuting a short position
                print('Executing short order',price,"with take profit",take_profit,"and stop loss",stop_loss)
                 


                self.exchange.create_order(
                        symbol=self.symbol,
                        type='market',
                        side='sell',
                        amount=qty
                    )

                self.exchange.create_order(
                    symbol=self.symbol,
                    type='stop',
                    side='buy',
                    amount=qty,
                    price=stop_loss,
                    params={
                        'stopPrice': stop_loss
                    }
                )

                self.exchange.create_order(
                    symbol=self.symbol,
                    type='take_profit_market',
                    side='buy',
                    amount=qty,
                    price=take_profit,
                    params={
                        'closePosition': True,
                        'stopPrice': take_profit
                    }
                )
                account = self.exchange.fetch_balance()['USDT']['Total']
                util.create_log(data.Date,price,'BUY',qty,account,0)
                    
                break
       