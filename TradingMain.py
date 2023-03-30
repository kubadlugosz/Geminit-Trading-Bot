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
    
    exchange = Client(api_key=config.key,api_secret=config.secret,tld='us',testnet=True)
    # #MACD
    symbol = "ETHUSDT"
    time_frame = '15m'
    df = util.getData(symbol,time_frame) 
    strategy = MyStrategy.MyStrategy()
    backtester = Backtester.Backtester(df, strategy,'RSI',1000, 500, 0.0099)
    
    print(df)



    # app = TradingApp()
    
    # # Start the Tk mainloop
    # app.mainloop()

   
main()





