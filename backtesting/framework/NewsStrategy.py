import backtrader as bt
from typing import Dict, Optional
from TradeLogic import TradeLogic

class NewsStrategy(bt.Strategy):
    params = dict(
        metadata=None,           # Event window metadata
        trade_signals=None,      # Pre-analyzed trade signals
        symbol='NVDA',          
        initial_cash=100000.0,
        max_position_size=100,
        max_exposure=0.2,
        trailing_stop_pct=0.02,
        max_drawdown_pct=0.05
    )
    
    def __init__(self):
        self.trade_logic = TradeLogic(
            initial_capital=self.p.initial_cash,
            max_position_size=self.p.max_position_size,
            max_exposure=self.p.max_exposure,
            trailing_stop_pct=self.p.trailing_stop_pct,
            max_drawdown_pct=self.p.max_drawdown_pct
        )
        
        self.order = None
        self.current_bar = 0
        self.signals = self.p.trade_signals or {}
        
    def next(self):
        # First update any existing positions
        if self.trade_logic.open_positions:
            current_prices = {self.p.symbol: self.data.close[0]}
            trades_to_execute = self.trade_logic.update_positions(
                current_prices, 
                self.data.datetime.datetime(0)
            )
            
            # Execute any position updates
            for trade in trades_to_execute:
                self.execute_trade(trade)
        
        # Check for signals at current timestamp
        current_time = self.data.datetime.datetime(0)
        if current_time in self.signals:
            signal = self.signals[current_time]
            self.process_signal(signal)
            
        self.current_bar += 1
        
    def process_signal(self, signal: Dict):
        """Process a pre-analyzed trade signal"""
        if not signal or self.order:
            return
            
        # Get trading decision from TradeLogic
        action, trade_details = self.trade_logic.process_news_analysis(
            signal,
            self.data.close[0],
            self.data.datetime.datetime(0).isoformat()
        )
        
        # Execute any resulting trades
        if trade_details:
            self.execute_trade(trade_details)
            
    def execute_trade(self, trade_details: Dict):
        """Execute a trade based on the provided details"""
        if self.order:
            return
            
        if trade_details['direction'] == 'BUY':
            self.order = self.buy(size=trade_details['size'])
            print(f"\nExecuting BUY: {trade_details['size']} shares @ ${trade_details['price']:.2f}")
        else:  # SELL
            self.order = self.sell(size=trade_details['size'])
            print(f"\nExecuting SELL: {trade_details['size']} shares @ ${trade_details['price']:.2f}")
            
    def get_metrics(self):
        """Return current trading metrics"""
        return self.trade_logic.get_metrics()