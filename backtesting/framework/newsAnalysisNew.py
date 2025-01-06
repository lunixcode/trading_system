# Standard library imports
import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Third-party imports
import backtrader as bt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Local imports
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator
from DataPreprocessor import DataPreprocessor
from newsAnalysisNew import NewsAnalysis
from trade_logic import TradeLogic  # New import for trade logic

class NewsStrategy(bt.Strategy):
    params = (
        ('news_data', None),
        ('news_indices', None),
        ('symbol', 'NVDA'),
        ('model_choice', 'gemini'),
        ('initial_cash', 100000.0),
        ('max_position_size', 100),
        ('max_exposure', 0.2),
        ('trailing_stop_pct', 0.02),
        ('max_drawdown_pct', 0.05)
    )
    
    def __init__(self):
        self.news_analyzer = NewsAnalysis(
            symbol=self.p.symbol,
            model_choice=self.p.model_choice
        )
        
        # Initialize TradeLogic
        self.trade_logic = TradeLogic(
            initial_capital=self.p.initial_cash,
            max_position_size=self.p.max_position_size,
            max_exposure=self.p.max_exposure,
            trailing_stop_pct=self.p.trailing_stop_pct,
            max_drawdown_pct=self.p.max_drawdown_pct
        )
        
        self.order = None
        self.news_time = None
        self.processed_indices = set()
        self.current_bar = 0
        self.news_indices = self.p.news_indices
        self.articles_processed = 0
        
        # Log initialization
        print(f"\nInitializing NewsStrategy:")
        print(f"Symbol: {self.p.symbol}")
        print(f"Model: {self.p.model_choice}")
    
    def next(self):
        # First, update existing positions with new prices
        if self.trade_logic.open_positions:
            current_prices = {
                self.p.symbol: self.data.close[0]
            }
            trades_to_execute = self.trade_logic.update_positions(
                current_prices, 
                self.data.datetime.datetime(0)
            )
            
            # Execute any position updates (e.g., trailing stop hits)
            for trade in trades_to_execute:
                self.execute_trade(trade)

        # Then process any new news
        current_datetime = self.data.datetime.datetime(0)
        
        if self.data.news_count[0] > 0:
            print(f"\nProcessing bar {self.current_bar} at {current_datetime}")
            print(f"News count: {self.data.news_count[0]}")
            
            if self.news_indices is not None:
                current_indices = self.news_indices.iloc[self.current_bar]
                if pd.notna(current_indices) and str(current_indices).strip():
                    self.process_news(current_indices, current_datetime)
        
        self.current_bar += 1

    def process_news(self, indices_str, current_datetime):
        """Process news and make trading decisions"""
        try:
            indices = [int(idx.strip()) for idx in str(indices_str).split(',') if idx.strip()]
            new_indices = set(indices) - self.processed_indices
            
            if new_indices:
                print(f"Found {len(new_indices)} new articles to analyze")
                
                for idx in new_indices:
                    if idx < len(self.p.news_data):
                        article = self.p.news_data[idx]
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                        analysis = loop.run_until_complete(
                            self.news_analyzer.analyze_news(article)
                        )
                        
                        if analysis:
                            self.log_analysis(analysis, current_datetime, article)
                            
                            # Convert LLM analysis to trade logic format
                            trade_analysis = self.convert_analysis_to_trade_format(analysis)
                            
                            # Get trading decision from TradeLogic
                            action, trade_details = self.trade_logic.process_news_analysis(
                                trade_analysis,
                                self.data.close[0],
                                current_datetime.isoformat()
                            )
                            
                            # Execute any resulting trades
                            if trade_details:
                                self.execute_trade(trade_details)
                        
                        self.articles_processed += 1
                        
                        if self.articles_processed % 5 == 0:
                            self.log_metrics()
                            input("Press Enter to continue with the next set of articles...")
                
                self.processed_indices.update(new_indices)
            
        except Exception as e:
            print(f"Error processing news indices: {e}")

    def convert_analysis_to_trade_format(self, analysis) -> Dict:
        """Convert LLM analysis to format expected by TradeLogic"""
        # Extract sentiment from impact score or detailed analysis
        sentiment = 0
        impact_score = float(analysis.get('impact_score', 0))
        
        if 'detailed_analysis' in analysis:
            detailed = analysis['detailed_analysis']
            if isinstance(detailed, dict):
                raw_sentiment = float(detailed.get('sentiment', 5))  # Default to neutral (5)
                # Convert 0-10 scale to -1 to 1 scale
                # 0 -> -0.8, 1 -> -0.6, 2 -> -0.4, ..., 5 -> 0, ..., 10 -> 1.0
                sentiment = (raw_sentiment - 5) * 0.2  # Each step is 0.2 (-0.8 to 1.0)
        
        # Log sentiment conversion
        if 'detailed_analysis' in analysis:
            raw_sentiment = float(analysis['detailed_analysis'].get('sentiment', 5))
            print(f"\nSentiment Conversion:")
            print(f"Raw sentiment (0-10): {raw_sentiment}")
            print(f"Converted sentiment (-0.8 to 1.0): {sentiment}")

        return {
            'symbol': self.p.symbol,
            'sentiment': sentiment,
            'confidence': abs(impact_score),  # Use impact score magnitude as confidence
            'impact': impact_score,
            'timestamp': datetime.now().isoformat()
        }

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

    def log_analysis(self, analysis: Dict, current_datetime: datetime, article: Dict):
        """Log the analysis results"""
        print("\n" + "="*50)
        print("News Analysis Results:")
        print("="*50)
        print(f"Time: {current_datetime}")
        print(f"Article: {article.get('title', 'No Title')}")
        print(f"Model Used: {self.p.model_choice}")
        print("-"*50)
        
        if 'initial_analysis' in analysis:
            print("\nInitial Analysis:")
            print(analysis['initial_analysis'])
        
        print(f"Impact Score: {analysis.get('impact_score', 'N/A')}")
        
        if analysis.get('detailed_analysis'):
            print("\nDetailed Analysis:")
            detailed = analysis['detailed_analysis']
            if isinstance(detailed, dict):
                if 'raw_analysis' in detailed:
                    print(detailed['raw_analysis'])
                else:
                    for key, value in detailed.items():
                        print(f"{key}: {value}")
        print("="*50 + "\n")

    def log_metrics(self):
        """Log current trading metrics"""
        metrics = self.trade_logic.get_metrics()
        print("\n" + "="*50)
        print("Trading Metrics:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Current Exposure: {metrics['current_exposure']*100:.2f}%")
        print(f"Return: {metrics['return_pct']*100:.2f}%")
        print("="*50 + "\n")