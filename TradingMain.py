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

#import python programs 
import Utilities as util
import MyStrategy
import Backtester



            

                        


def main():
    
    #exchange = Client(api_key=config.key,api_secret=config.secret,tld='us',testnet=True)
    # #MACD
<<<<<<< HEAD
    symbol = "LINKUSDT"
    time_frame = '5m'
=======
    symbol = "LINK/USDT"
    time_frame = '15m'
>>>>>>> cbadb8e62a48e5a3a0f9b23302ef716621ced3db
    df = util.getData(symbol,time_frame) 
    strategy = MyStrategy.MyStrategy()
    backtester = Backtester.Backtester(df, strategy,'LinearRegression',1000, 500, 0.0099)
    
<<<<<<< HEAD
    params = {'k_period': 14, 'd_period': 3,'vzo_length':14}
=======
    params = {'k_period': 14, 'd_period': 3}
>>>>>>> cbadb8e62a48e5a3a0f9b23302ef716621ced3db
    #signals = strategy.generate_signals_backtest(df,'LinearRegression',**params)
    
    backtester.run_backtest(**params)

    # app = TradingApp()
    
    # # Start the Tk mainloop
    # app.mainloop()

   
main()





