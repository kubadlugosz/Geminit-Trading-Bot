
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