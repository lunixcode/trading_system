# Standard library imports
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime

# Third-party imports
import pandas as pd
import numpy as np

@dataclass
class TradeMetrics:
    max_drawdown: float = 0.0
    trailing_stop: float = 0.0
    position_size: int = 0
    current_exposure: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0

class TradeLogic:
    def __init__(self, 
                 initial_capital: float,
                 max_position_size: int = 100,
                 max_exposure: float = 0.2,  # 20% max portfolio exposure
                 trailing_stop_pct: float = 0.02,  # 2% trailing stop
                 max_drawdown_pct: float = 0.05):  # 5% max drawdown
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_exposure = max_exposure
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        self.metrics = TradeMetrics()
        self.open_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
    def process_news_analysis(self, news_analysis: Dict, current_price: float, timestamp: str) -> Tuple[str, Optional[Dict]]:
        """Process news analysis and return trading decision"""
        # Extract sentiment and confidence from news analysis
        sentiment = news_analysis.get('sentiment', 0)  # -0.8 to 1.0 scale
        confidence = news_analysis.get('confidence', 0)  # Impact score magnitude
        impact = news_analysis.get('impact', 0)
        
        # Combined score for decision making
        score = sentiment * confidence * impact
        
        # Determine position size based on confidence and current exposure
        potential_position_size = self._calculate_position_size(score, current_price)
        
        # Strong signal threshold (may need tuning)
        if abs(score) > 0.7:
            if score > 0 and self._can_open_long(current_price, potential_position_size):
                return self._open_long_position(current_price, potential_position_size, news_analysis, timestamp)
            elif score < 0 and self._can_open_short(current_price, potential_position_size):
                return self._open_short_position(current_price, potential_position_size, news_analysis, timestamp)
                
        return "HOLD", None

    def update_positions(self, current_prices: Dict[str, float], timestamp: str) -> List[Dict]:
        """Update position states and check for closing conditions"""
        trades_to_execute = []
        
        for symbol, position in list(self.open_positions.items()):
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Update trailing stop
            if position['direction'] == 'LONG':
                if current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    position['trailing_stop'] = current_price * (1 - self.trailing_stop_pct)
                
                if current_price <= position['trailing_stop']:
                    trades_to_execute.append(
                        self._close_position(symbol, current_price, 'TRAILING_STOP', timestamp)
                    )
                    
            else:  # SHORT position
                if current_price < position['lowest_price']:
                    position['lowest_price'] = current_price
                    position['trailing_stop'] = current_price * (1 + self.trailing_stop_pct)
                    
                if current_price >= position['trailing_stop']:
                    trades_to_execute.append(
                        self._close_position(symbol, current_price, 'TRAILING_STOP', timestamp)
                    )
                    
        return trades_to_execute

    def _calculate_position_size(self, score: float, current_price: float) -> int:
        """Calculate position size based on score and risk parameters"""
        max_capital_exposure = self.current_capital * self.max_exposure
        max_shares = int(max_capital_exposure / current_price)
        
        # Scale position size by score magnitude (0.7 to 1.0)
        score_magnitude = min(abs(score), 1.0)
        scaled_size = int(max_shares * ((score_magnitude - 0.7) / 0.3))
        
        return min(scaled_size, self.max_position_size)

    def _can_open_long(self, price: float, size: int) -> bool:
        """Check if we can open a long position"""
        potential_exposure = (price * size) / self.current_capital
        return (potential_exposure + self._get_current_exposure()) <= self.max_exposure

    def _can_open_short(self, price: float, size: int) -> bool:
        """Check if we can open a short position"""
        return self._can_open_long(price, size)  # Same logic for now

    def _open_long_position(self, price: float, size: int, analysis: Dict, timestamp: str) -> Tuple[str, Dict]:
        """Open a long position"""
        position = {
            'direction': 'LONG',
            'size': size,
            'entry_price': price,
            'highest_price': price,
            'trailing_stop': price * (1 - self.trailing_stop_pct),
            'entry_time': timestamp,
            'analysis': analysis
        }
        
        symbol = analysis.get('symbol', 'UNKNOWN')
        self.open_positions[symbol] = position
        
        return "BUY", {
            'symbol': symbol,
            'size': size,
            'price': price,
            'timestamp': timestamp,
            'reason': 'NEWS_SIGNAL'
        }

    def _open_short_position(self, price: float, size: int, analysis: Dict, timestamp: str) -> Tuple[str, Dict]:
        """Open a short position"""
        position = {
            'direction': 'SHORT',
            'size': size,
            'entry_price': price,
            'lowest_price': price,
            'trailing_stop': price * (1 + self.trailing_stop_pct),
            'entry_time': timestamp,
            'analysis': analysis
        }
        
        symbol = analysis.get('symbol', 'UNKNOWN')
        self.open_positions[symbol] = position
        
        return "SELL", {
            'symbol': symbol,
            'size': size,
            'price': price,
            'timestamp': timestamp,
            'reason': 'NEWS_SIGNAL'
        }

    def _close_position(self, symbol: str, price: float, reason: str, timestamp: str) -> Dict:
        """Close a position and update metrics"""
        position = self.open_positions[symbol]
        pnl = self._calculate_pnl(position, price)
        
        trade_record = {
            'symbol': symbol,
            'direction': position['direction'],
            'size': position['size'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'pnl': pnl,
            'reason': reason,
            'analysis': position['analysis']
        }
        
        self.trade_history.append(trade_record)
        self._update_metrics(trade_record)
        del self.open_positions[symbol]
        
        return {
            'symbol': symbol,
            'size': position['size'],
            'price': price,
            'timestamp': timestamp,
            'reason': reason,
            'direction': 'BUY' if position['direction'] == 'SHORT' else 'SELL'
        }

    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate PnL for a position"""
        if position['direction'] == 'LONG':
            return (current_price - position['entry_price']) * position['size']
        else:  # SHORT
            return (position['entry_price'] - current_price) * position['size']

    def _update_metrics(self, trade_record: Dict) -> None:
        """Update trading metrics after closing a position"""
        self.metrics.total_trades += 1
        if trade_record['pnl'] > 0:
            self.metrics.winning_trades += 1
        else:
            self.metrics.losing_trades += 1
            
        self.metrics.total_pnl += trade_record['pnl']
        self.current_capital += trade_record['pnl']
        
        # Update max drawdown
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        self.metrics.max_drawdown = max(self.metrics.max_drawdown, drawdown)

    def _get_current_exposure(self) -> float:
        """Calculate current total exposure"""
        total_exposure = sum(
            abs(pos['size'] * pos['entry_price']) 
            for pos in self.open_positions.values()
        )
        return total_exposure / self.current_capital

    def get_metrics(self) -> Dict:
        """Return current trading metrics"""
        return {
            'total_trades': self.metrics.total_trades,
            'winning_trades': self.metrics.winning_trades,
            'losing_trades': self.metrics.losing_trades,
            'win_rate': self.metrics.winning_trades / max(1, self.metrics.total_trades),
            'total_pnl': self.metrics.total_pnl,
            'max_drawdown': self.metrics.max_drawdown,
            'current_exposure': self._get_current_exposure(),
            'current_capital': self.current_capital,
            'return_pct': (self.current_capital - self.initial_capital) / self.initial_capital
        }