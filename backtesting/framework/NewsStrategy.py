from datetime import datetime
from typing import Dict
import backtrader as bt
import pandas as pd
from TradeLogic import TradeLogic

class NewsStrategy(bt.Strategy):
    params = (
        ('news_data', None),          # Pre-analyzed news data
        ('news_indices', None),       # News indices for time alignment
        ('symbol', 'NVDA'),          
        ('initial_cash', 100000.0),
        ('max_position_size', 100),
        ('max_exposure', 0.2),
        ('trailing_stop_pct', 0.02),
        ('max_drawdown_pct', 0.05)
    )
    
    def __init__(self):
        # Initialize TradeLogic
        self.trade_logic = TradeLogic(
            initial_capital=self.p.initial_cash,
            max_position_size=self.p.max_position_size,
            max_exposure=self.p.max_exposure,
            trailing_stop_pct=self.p.trailing_stop_pct,
            max_drawdown_pct=self.p.max_drawdown_pct
        )
        
        self.order = None
        self.current_bar = 0
        self.news_indices = self.p.news_indices
        self.processed_indices = set()
    
    def next(self):
        # First update any existing positions
        if self.trade_logic.open_positions:
            current_prices = {self.p.symbol: self.data.close[0]}
            trades_to_execute = self.trade_logic.update_positions(
                current_prices, 
                self.data.datetime.datetime(0)
            )
            
            # Execute any position updates (e.g., trailing stop hits)
            for trade in trades_to_execute:
                self.execute_trade(trade)

        # Process any new news signals
        if self.data.news_count[0] > 0:
            current_indices = self.news_indices.iloc[self.current_bar]
            if pd.notna(current_indices) and str(current_indices).strip():
                self.process_news_signals(current_indices)
        
        self.current_bar += 1

    def process_news_signals(self, indices_str: str):
        """Process pre-analyzed news signals"""
        try:
            indices = [int(idx.strip()) for idx in str(indices_str).split(',') if idx.strip()]
            new_indices = set(indices) - self.processed_indices
            
            if new_indices:
                for idx in new_indices:
                    if idx < len(self.p.news_data):
                        # Get pre-analyzed news data
                        analysis = self.p.news_data[idx]
                        
                        # Get trading decision from TradeLogic
                        action, trade_details = self.trade_logic.process_news_analysis(
                            analysis,
                            self.data.close[0],
                            self.data.datetime.datetime(0).isoformat()
                        )
                        
                        # Execute any resulting trades
                        if trade_details:
                            self.execute_trade(trade_details)
                
                self.processed_indices.update(new_indices)
            
        except Exception as e:
            print(f"Error processing news signals: {e}")

    def execute_trade(self, trade_details: Dict):
        """Execute a trade based on the provided details"""
        if self.order:
            return  # Already waiting for order to complete
            
        if trade_details['direction'] == 'BUY':
            self.order = self.buy(size=trade_details['size'])
            print(f"\nPlacing BUY order for {trade_details['size']} shares at {trade_details['price']}")
        else:  # SELL
            self.order = self.sell(size=trade_details['size'])
            print(f"\nPlacing SELL order for {trade_details['size']} shares at {trade_details['price']}")

    def get_metrics(self):
        """Return current trading metrics"""
        return self.trade_logic.get_metrics()