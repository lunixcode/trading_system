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
from newsAnalysisNew import newsAnalysisNew

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
    
    print("\nBefore date processing:")
    print(df.index[:5])
    print(df['Date'].head() if 'Date' in df.columns else "No Date column")
    
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
    
    print("\nAfter date processing:")
    print(df.index[:5])
    
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
    
    print("\nFinal data check:")
    print("Index type:", type(df.index))
    print("First few dates:", df.index[:5])
    
    return df

class NewsStrategy(bt.Strategy):
    params = (
        ('news_data', None),
        ('news_indices', None),
    )
    
    def __init__(self):
        self.news_analyzer = newsAnalysisNew(symbol="NVDA")
        self.order = None
        self.news_time = None
        self.processed_indices = set()
        self.current_bar = 0
        self.news_indices = self.p.news_indices
    
    def next(self):
        current_datetime = self.data.datetime.datetime(0)
        
        # Check for news
        if self.data.news_count[0] > 0:
            print(f"\nProcessing bar {self.current_bar} at {current_datetime}")
            print(f"News count: {self.data.news_count[0]}")
            
            # Get news indices for current bar if available
            if self.news_indices is not None:
                current_indices = self.news_indices.iloc[self.current_bar]
                if pd.notna(current_indices) and str(current_indices).strip():
                    self.process_news(current_indices, current_datetime)
        
        self.current_bar += 1
    
    def process_news(self, indices_str, current_datetime):
        """Process news for current bar"""
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
                            print("\nNews Analysis Results:")
                            print(f"Time: {current_datetime}")
                            print(f"Article: {article.get('title', 'No Title')}")
                            print(f"Analysis: {analysis}")
                
                self.processed_indices.update(new_indices)
            
        except Exception as e:
            print(f"Error processing news indices: {e}")

def run_backtest(price_data_df, news_data, **kwargs):
    """Run backtest with fixed data types"""
    cerebro = bt.Cerebro()
    
    print("\nPreparing data...")
    prepared_data = prepare_backtest_data(price_data_df)
    
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
        news_indices=prepared_data.attrs.get('news_indices')
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
        # Initialize components
        hdm = HistoricalDataManager()
        validator = DataValidator()
        preprocessor = DataPreprocessor()
        
        # Set date range
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        symbol = "NVDA"
        
        print("Loading data...")
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
            initial_cash=100000.0
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error("Stack trace:", exc_info=True)
        sys.exit(1)