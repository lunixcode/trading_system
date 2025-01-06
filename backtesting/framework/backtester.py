# Standard library imports
import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Third-party imports
import backtrader as bt
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv


# Local imports
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator
from DataPreprocessor import DataPreprocessor
from newsAnalysisNew import NewsAnalysis
from TradeLogic import TradeLogic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PandasNewsData(bt.feeds.PandasData):
    """Custom data feed for price data with news information"""
    lines = ('news_count',)
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('news_count', 'news_count'),
    )

def prepare_backtest_data(df):
    """Prepare DataFrame for Backtrader with correct date handling"""
    df = df.copy()
    
    #print("\nBefore date processing:")
    #print(df.index[:5])
    #print(df['Date'].head() if 'Date' in df.columns else "No Date column")
    
    # Proper date handling
    if 'Date' in df.columns:
        # Convert Date column to datetime if it isn't already
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Set the Date column as index
        df.set_index('Date', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # If no Date column but index isn't datetime, try to convert index
        df.index = pd.to_datetime(df.index)
    
    #print("\nAfter date processing:")
    #print(df.index[:5])
    
    # Store news indices separately
    news_indices = None
    if 'news_indices' in df.columns:
        news_indices = df['news_indices'].copy()
        df = df.drop('news_indices', axis=1)
    
    # Convert numeric columns
    numeric_columns = {
        'Open': 'float64',
        'High': 'float64',
        'Low': 'float64',
        'Close': 'float64',
        'Volume': 'float64',
        'news_count': 'int32'
    }
    
    for col, dtype in numeric_columns.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
    
    # Fill any NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Store news indices in metadata
    df.attrs['news_indices'] = news_indices
    
    #print("\nFinal data check:")
    #print("Index type:", type(df.index))
    #print("First few dates:", df.index[:5])
    
    return df

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
                        
                        print("\n" + "="*50)
                        print("Processing News Article:")
                        print(f"Index: {idx}")
                        print(f"Raw News Data: {self.p.news_data[idx]}")  # Added this line
                        print(f"Initial Analysis: {analysis.get('initial_analysis', 'No initial analysis')}")
                        print(f"Impact Score: {analysis.get('impact_score', 0)}")
                        print(f"Has Detailed Analysis: {'Yes' if analysis.get('detailed_analysis') else 'No'}")
                        print("="*50 + "\n")
                        
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

def run_backtest(price_data_df, news_data, model_choice='gpt', **kwargs):
    """Run backtest with fixed data types"""
    cerebro = bt.Cerebro()
    
    print(f"\nPreparing data for backtest using {model_choice} model...")
    prepared_data = prepare_backtest_data(price_data_df)
    
    # Debug news data
    print("\nChecking news data:")
    print(f"Number of news items: {len(news_data)}")
    print("Sample news item:", news_data[0] if news_data else "No news data")
    

    # Create data feed without news_indices
    data_feed = PandasNewsData(
        dataname=prepared_data,
        fromdate=prepared_data.index.min(),
        todate=prepared_data.index.max()
    )
    
    cerebro.adddata(data_feed)
    
    # Initialize strategy with both the news data and indices
    cerebro.addstrategy(
        NewsStrategy,
        news_data=news_data,
        news_indices=prepared_data.attrs.get('news_indices'),
        symbol=kwargs.get('symbol', 'NVDA')
    )
    
    # Set initial capital
    cerebro.broker.setcash(kwargs.get('initial_cash', 100000.0))
    
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    
    try:
        print("\nRunning backtest...")
        runner = cerebro.run()
        print("Backtest completed successfully")
        
        final_value = cerebro.broker.getvalue()
        print(f'Final Portfolio Value: ${final_value:.2f}')
        
        return runner
        
    except Exception as e:
        print(f"\nError during backtest: {str(e)}")
        print("\nData sample where error occurred:")
        print(prepared_data.head())
        raise

if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("News Analysis Backtesting System")
        print("="*50)

        # Model selection
        print("\nAvailable Models:")
        print("1. GPT-4 (gpt)")
        print("2. Gemini Pro (gemini)")
        print("3. Claude 3 (claude)")
        print("4. Together AI (together)")
        
        while True:
            model_choice = input("\nSelect model (enter the name in brackets): ").lower().strip()
            if model_choice in ['gpt', 'gemini', 'claude', 'together']:
                break
            print("Invalid choice. Please select from the available models.")

        # Stock selection
        print("\nPopular Stocks:")
        print("1. NVIDIA (NVDA)")
        print("2. Apple (AAPL)")
        print("3. Microsoft (MSFT)")
        print("4. Tesla (TSLA)")
        print("5. Meta (META)")
        print("Or enter any other valid stock ticker")
        
        symbol = input("\nEnter stock symbol: ").upper().strip()

        print(f"\nInitializing backtest for {symbol} using {model_choice} model...")
        
        # Initialize components
        hdm = HistoricalDataManager()
        validator = DataValidator()
        preprocessor = DataPreprocessor()
        
        # Set date range
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        print("\nLoading data...")
        # Get and prepare data
        price_data = hdm.get_price_data(symbol, start_date, end_date)
        news_data = hdm.get_news_data(symbol, start_date, end_date)
        
        print("\nValidating news data...")
        _, _, valid_news = validator.validate_news_data(news_data)
        
        print("\nProcessing and aligning data...")
        aligned_data, processed_news = preprocessor.align_all_timeframes(price_data, valid_news)
        
        print("\nStarting backtest...")
        results = run_backtest(
            price_data_df=aligned_data['5min'],
            news_data=processed_news,
            model_choice=model_choice,
            initial_cash=100000.0,
            symbol=symbol
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error("Stack trace:", exc_info=True)
        sys.exit(1)