import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backtesting.framework.DataPreprocessor import DataPreprocessor

@dataclass
class PriceMove:
    """Represents a significant price movement with associated data."""
    date: datetime
    timeframe: str
    percentage_change: float
    absolute_change: float
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: float
    news_count: int
    news_indices: str

class StatisticalAnalyzer:
    """
    Analyzes price movements and identifies significant market moves.
    Focuses on pure statistical analysis without LLM integration.
    """
    
    def __init__(self, cache_dir: str = "cache", debug: bool = False):
        """Initialize the StatisticalAnalyzer."""
        self.debug = debug
        self.cache_dir = Path(cache_dir)
        
    def calculate_price_moves(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various price movement metrics for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional columns for price movement metrics
        """
        moves_df = df.copy()
        
        # Calculate absolute and percentage changes
        moves_df['absolute_change'] = moves_df['Close'] - moves_df['Open']
        moves_df['percentage_change'] = ((moves_df['Close'] - moves_df['Open']) / moves_df['Open']) * 100
        
        # Calculate true range using vectorized operations
        high_low = moves_df['High'] - moves_df['Low']
        high_close_prev = abs(moves_df['High'] - moves_df['Close'].shift(1))
        low_close_prev = abs(moves_df['Low'] - moves_df['Close'].shift(1))
        
        # Combine all components
        moves_df['true_range'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Handle first row where we don't have previous close
        moves_df.loc[moves_df.index[0], 'true_range'] = high_low.iloc[0]
        
        return moves_df

    def get_largest_moves(self, 
                         preprocessor: 'DataPreprocessor',
                         timeframe: str,
                         n_moves: int = 10,
                         by_percentage: bool = True) -> List[PriceMove]:
        """
        Find the n largest price moves in the dataset for a given timeframe.
        
        Args:
            preprocessor: DataPreprocessor instance with aligned data
            timeframe: Timeframe to analyze ('5min', '15min', '30min', '1h', '1d')
            n_moves: Number of largest moves to return
            by_percentage: If True, sort by percentage change; if False, sort by absolute change
            
        Returns:
            List of PriceMove objects representing the largest moves
        """
        if timeframe not in preprocessor.aligned_data:
            raise ValueError(f"No data available for timeframe {timeframe}")
            
        if self.debug:
            print(f"\nAnalyzing largest moves for {timeframe} timeframe")
            
        # Get data for the specified timeframe
        df = preprocessor.aligned_data[timeframe]
        
        # Calculate moves
        moves_df = self.calculate_price_moves(df)
        
        # Sort by specified metric
        sort_column = 'percentage_change' if by_percentage else 'absolute_change'
        # Create absolute value column for sorting
        moves_df['abs_change'] = moves_df[sort_column].abs()
        largest_moves = moves_df.nlargest(n_moves, 'abs_change')
        # Drop the temporary column
        largest_moves = largest_moves.drop('abs_change', axis=1)
        
        # Convert to PriceMove objects
        price_moves = []
        for _, row in largest_moves.iterrows():
            move = PriceMove(
                date=row['Date'],
                timeframe=timeframe,
                percentage_change=row['percentage_change'],
                absolute_change=row['absolute_change'],
                open_price=row['Open'],
                close_price=row['Close'],
                high_price=row['High'],
                low_price=row['Low'],
                volume=row['Volume'],
                news_count=row['news_count'],
                news_indices=row['news_indices'] if pd.notna(row['news_indices']) else ''
            )
            price_moves.append(move)
            
        if self.debug:
            print(f"\nFound {len(price_moves)} significant price moves")
            for move in price_moves:
                print(f"\nDate: {move.date}")
                print(f"{'Percentage' if by_percentage else 'Absolute'} Change: "
                      f"{move.percentage_change:.2f}% (${move.absolute_change:.2f})")
                print(f"News Count: {move.news_count}")
                
        return price_moves

    def get_news_context(self, 
                        preprocessor: 'DataPreprocessor', 
                        move: PriceMove,
                        lookback_periods: int = 3) -> List[dict]:
        """
        Get news context for a specific price move.
        
        Args:
            preprocessor: DataPreprocessor instance with aligned data
            move: PriceMove object to analyze
            lookback_periods: Number of periods to look back for news
            
        Returns:
            List of news items with temporal relationship to the move
        """
        if self.debug:
            print(f"\nGetting news context for move on {move.date}")
            
        # Get news for the specific period
        news_items = []
        if move.news_indices and move.news_indices.strip():
            current_indices = [int(idx) for idx in move.news_indices.split(',')]
            for idx in current_indices:
                if idx < len(preprocessor.news_data):
                    news_item = preprocessor.news_data[idx].copy()
                    news_item['temporal_relationship'] = 'during_move'
                    news_items.append(news_item)
        
        # Get data for the lookback period
        df = preprocessor.aligned_data[move.timeframe]
        # Convert move.date to tz-naive for comparison
        move_date_naive = pd.to_datetime(move.date).tz_localize(None)
        df_dates_naive = df['Date'].dt.tz_localize(None)
        move_idx = df[df_dates_naive == move_date_naive].index[0]
        
        # Look back several periods
        for i in range(1, lookback_periods + 1):
            if move_idx - i >= 0:
                prior_row = df.iloc[move_idx - i]
                if prior_row['news_count'] > 0 and pd.notna(prior_row['news_indices']):
                    prior_indices = [int(idx) for idx in prior_row['news_indices'].split(',')]
                    for idx in prior_indices:
                        if idx < len(preprocessor.news_data):
                            news_item = preprocessor.news_data[idx].copy()
                            news_item['temporal_relationship'] = f'{i}_periods_before'
                            # Convert both datetimes to tz-naive for comparison
                            move_date = pd.to_datetime(move.date).tz_localize(None)
                            news_date = pd.to_datetime(news_item['date']).tz_localize(None)
                            news_item['time_to_move'] = move_date - news_date
                            news_items.append(news_item)
        
        return news_items

def main():
    """Example usage of the StatisticalAnalyzer"""
    from backtesting.framework.HistoricalDataManager import HistoricalDataManager
    from backtesting.framework.DataValidator import DataValidator
    from backtesting.framework.DataPreprocessor import DataPreprocessor
    
    # Initialize components
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    preprocessor = DataPreprocessor(cache_dir="cache", debug=True)
    analyzer = StatisticalAnalyzer(debug=True)
    
    # Load and process data
    symbol = 'NVDA'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    # Load initial data
    price_data = hdm.get_price_data(symbol, start_date, end_date)
    raw_news_data = hdm.get_news_data(symbol, start_date, end_date)
    _, _, valid_news_data = validator.validate_news_data(raw_news_data)
    
    # Process all timeframes with caching
    preprocessor.align_all_timeframes(price_data, valid_news_data, symbol)
    
    # Get significant moves
    print("\nAnalyzing Daily Moves:")
    daily_moves = analyzer.get_largest_moves(preprocessor, '1d', n_moves=10)
    
    print("\nAnalyzing Hourly Moves:")
    hourly_moves = analyzer.get_largest_moves(preprocessor, '1h', n_moves=10)
    
    # Print some sample news context
    if daily_moves:
        print("\nSample News Context for First Major Move:")
        news_context = analyzer.get_news_context(preprocessor, daily_moves[0])
        for news in news_context:
            print(f"\nTitle: {news['title']}")
            print(f"Temporal Relationship: {news['temporal_relationship']}")

if __name__ == "__main__":
    main()