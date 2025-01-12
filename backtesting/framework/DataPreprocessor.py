from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path

class DataPreprocessor:
    def __init__(self, cache_dir: str = "cache", debug: bool = False):
        self.debug = debug
        self.timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': 'h',
            '1d': 'D'
        }
        self.cache_dir = Path(cache_dir)
        self.aligned_data = {}
        self.news_data = None

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.debug:
            print(f"Cache directory: {self.cache_dir}")

    def _get_cache_path(self, symbol: str, year: int) -> Path:
        """Get the cache directory path for a symbol and year."""
        return self.cache_dir / str(symbol) / str(year)

    def _save_to_cache(self, symbol: str, start_date: datetime) -> None:
        """Save aligned data and news data to disk cache."""
        if self.debug:
            print("\nSaving data to cache...")
        
        cache_path = self._get_cache_path(symbol, start_date.year)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Save aligned data for each timeframe
        for timeframe, df in self.aligned_data.items():
            cache_file = cache_path / f"aligned_{timeframe}.pkl"
            df.to_pickle(str(cache_file))
            if self.debug:
                print(f"Saved {timeframe} data to {cache_file}")

        # Save news data
        if self.news_data:
            news_cache_file = cache_path / "news_data.pkl"
            with open(news_cache_file, 'wb') as f:
                pickle.dump(self.news_data, f)
            if self.debug:
                print(f"Saved news data to {news_cache_file}")

    def _load_from_cache(self, symbol: str, year: int) -> bool:
        """Load aligned data and news data from disk cache."""
        if self.debug:
            print("\nAttempting to load data from cache...")
        
        cache_path = self._get_cache_path(symbol, year)
        
        if not cache_path.exists():
            if self.debug:
                print("No cache directory found")
            return False

        try:
            # Load aligned data for each timeframe
            self.aligned_data = {}
            for timeframe in self.timeframes:
                cache_file = cache_path / f"aligned_{timeframe}.pkl"
                if cache_file.exists():
                    self.aligned_data[timeframe] = pd.read_pickle(str(cache_file))
                    if self.debug:
                        print(f"Loaded {timeframe} data from cache")

            # Load news data
            news_cache_file = cache_path / "news_data.pkl"
            if news_cache_file.exists():
                with open(news_cache_file, 'rb') as f:
                    self.news_data = pickle.load(f)
                if self.debug:
                    print("Loaded news data from cache")

            return True if self.aligned_data and self.news_data else False

        except Exception as e:
            if self.debug:
                print(f"Error loading from cache: {str(e)}")
            return False

    def resample_price_data(self, price_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample price data to specified timeframe."""
        if self.debug:
            print(f"\nResampling price data to {timeframe} timeframe")
            print(f"Original shape: {price_data.shape}")

        df = price_data.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)

        freq = self.timeframes[timeframe]
        resampled = pd.DataFrame()
        resampled['Open'] = df['Open'].resample(freq).first()
        resampled['High'] = df['High'].resample(freq).max()
        resampled['Low'] = df['Low'].resample(freq).min()
        resampled['Close'] = df['Close'].resample(freq).last()
        resampled['Volume'] = df['Volume'].resample(freq).sum()

        resampled.reset_index(inplace=True)
        
        if self.debug:
            print(f"Resampled shape: {resampled.shape}")

        return resampled

    def align_price_and_news(self, price_data: pd.DataFrame, news_data: List[dict], timeframe: str) -> Tuple[pd.DataFrame, List[dict]]:
        """Align news data with price data for a specific timeframe."""
        if self.debug:
            print(f"\nAligning data for {timeframe} timeframe...")

        # Resample price data
        resampled_price = self.resample_price_data(price_data, timeframe)
        
        # Convert news data to DataFrame
        news_df = pd.DataFrame(news_data)
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Extract date components for both datasets
        for df in [news_df, resampled_price]:
            df['year'] = df['date' if 'date' in df else 'Date'].dt.year
            df['month'] = df['date' if 'date' in df else 'Date'].dt.month
            df['day'] = df['date' if 'date' in df else 'Date'].dt.day
            df['hour'] = df['date' if 'date' in df else 'Date'].dt.hour
            df['minute'] = df['date' if 'date' in df else 'Date'].dt.minute
        
        news_df['news_index'] = news_df.index

        # Create aligned DataFrame
        aligned_data = resampled_price.copy()
        aligned_data['news_count'] = 0
        aligned_data['news_indices'] = ''

        # Match based on timeframe
        for idx, news_item in news_df.iterrows():
            try:
                if timeframe == '1d':
                    mask = ((aligned_data['year'] == news_item['year']) & 
                           (aligned_data['month'] == news_item['month']) & 
                           (aligned_data['day'] == news_item['day']))
                else:
                    # For intraday timeframes, round minutes based on timeframe
                    news_minutes = news_item['minute']
                    if timeframe == '5min':
                        period_minutes = (news_minutes // 5) * 5
                    elif timeframe == '15min':
                        period_minutes = (news_minutes // 15) * 15
                    elif timeframe == '30min':
                        period_minutes = (news_minutes // 30) * 30
                    else:  # 1h
                        period_minutes = 0
                    
                    mask = ((aligned_data['year'] == news_item['year']) & 
                           (aligned_data['month'] == news_item['month']) & 
                           (aligned_data['day'] == news_item['day']) & 
                           (aligned_data['hour'] == news_item['hour']))
                    
                    if timeframe != '1h':
                        mask = mask & (aligned_data['minute'] == period_minutes)
                
                if mask.any():
                    match_idx = mask.idxmax()
                    aligned_data.loc[match_idx, 'news_count'] += 1
                    
                    # Add news index
                    current_indices = aligned_data.loc[match_idx, 'news_indices']
                    new_index = str(news_item['news_index'])
                    if current_indices:
                        aligned_data.loc[match_idx, 'news_indices'] += f",{new_index}"
                    else:
                        aligned_data.loc[match_idx, 'news_indices'] = new_index
                    
            except Exception as e:
                if self.debug:
                    print(f"Error matching news item {idx}: {e}")
                continue

        # Clean up temporary columns
        aligned_data.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1, inplace=True)

        if self.debug:
            print(f"Alignment complete for {timeframe}")
            print(f"Total periods: {len(aligned_data)}")
            print(f"Periods with news: {(aligned_data['news_count'] > 0).sum()}")

        return aligned_data, news_data

    def align_all_timeframes(self, price_data: pd.DataFrame, news_data: List[dict], symbol: str) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
        """Align data for all timeframes with disk caching."""
        if self.debug:
            print("\nProcessing all timeframes...")
        
        start_date = pd.to_datetime(price_data['Date'].min())
        end_date = pd.to_datetime(price_data['Date'].max())
        
        # Try to load from cache first
        if self._load_from_cache(symbol, start_date.year):
            try:
                # Get date ranges from each timeframe's cached data
                date_ranges = []
                for timeframe, df in self.aligned_data.items():
                    df_start = pd.to_datetime(df['Date'].min())
                    df_end = pd.to_datetime(df['Date'].max())
                    date_ranges.append((df_start, df_end))
                    if self.debug:
                        print(f"{timeframe} cache range: {df_start} to {df_end}")
                
                # Check if cache covers the requested period for all timeframes
                cache_valid = all(
                    abs((df_start - start_date).days) <= 1 and  # Within 1 day tolerance
                    abs((df_end - end_date).days) <= 1
                    for df_start, df_end in date_ranges
                )
                
                if cache_valid:
                    if self.debug:
                        print("Using cache - date ranges match")
                    return self.aligned_data, self.news_data
                else:
                    if self.debug:
                        print(f"Cache date mismatch - Requested: {start_date} to {end_date}")
            except Exception as e:
                if self.debug:
                    print(f"Error checking cache dates: {e}")
        
        if self.debug:
            print("Processing new alignment")
        
        self.news_data = news_data
        
        # Process each timeframe
        for timeframe in self.timeframes.keys():
            if self.debug:
                print(f"\nProcessing {timeframe} timeframe...")
            
            aligned_df, _ = self.align_price_and_news(price_data, news_data, timeframe)
            self.aligned_data[timeframe] = aligned_df
        
        # Save to cache
        self._save_to_cache(symbol, start_date)
        
        return self.aligned_data, self.news_data

    def get_data_for_date(self, date: str, timeframe: str = '1d') -> Tuple[pd.DataFrame, List[dict]]:
        """Get all data for a specific date and timeframe."""
        if timeframe not in self.aligned_data:
            raise ValueError(f"No data available for timeframe {timeframe}")
            
        df = self.aligned_data[timeframe]
        target_dt = pd.to_datetime(date)
        
        if timeframe == '1d':
            mask = df['Date'].dt.date == target_dt.date()
        else:
            mask = df['Date'].dt.date == target_dt.date()
            
        return df[mask], self.news_data

    def get_news_for_period(self, date: str, timeframe: str = '1d') -> List[dict]:
        """Get news items for a specific period."""
        df, news_data = self.get_data_for_date(date, timeframe)
        
        if df.empty:
            return []
            
        news_items = []
        for _, row in df.iterrows():
            if row['news_count'] > 0:
                indices = [int(idx) for idx in row['news_indices'].split(',')]
                period_news = [news_data[idx] for idx in indices]
                news_items.extend(period_news)
                
        return news_items

def main():
    """Load and process data for 2024"""
    from datetime import datetime
    from HistoricalDataManager import HistoricalDataManager
    from DataValidator import DataValidator
    
    # Initialize components
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    preprocessor = DataPreprocessor(debug=True)
    
    # Set date range for 2024
    symbol = 'NVDA'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Load and process data
    price_data = hdm.get_price_data(symbol, start_date, end_date)
    raw_news_data = hdm.get_news_data(symbol, start_date, end_date)
    _, _, valid_news_data = validator.validate_news_data(raw_news_data)
    
    # Align data (will use cache if available)
    preprocessor.align_all_timeframes(price_data, valid_news_data, symbol)

if __name__ == "__main__":
    main()