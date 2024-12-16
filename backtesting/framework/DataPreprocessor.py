from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataPreprocessor:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': 'h',
            '1d': 'D'
        }
        # Store aligned data in memory
        self.aligned_data = {}
        self.news_data = None

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

    def align_all_timeframes(self, price_data: pd.DataFrame, news_data: List[dict]) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
        """Align data for all timeframes."""
        if self.debug:
            print("\nProcessing all timeframes...")
        
        self.news_data = news_data  # Store news data
        
        for timeframe in self.timeframes.keys():
            if self.debug:
                print(f"\nProcessing {timeframe} timeframe...")
            
            aligned_df, _ = self.align_price_and_news(price_data, news_data, timeframe)
            self.aligned_data[timeframe] = aligned_df
        
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
    """Example usage of the DataPreprocessor"""
    from datetime import datetime
    from HistoricalDataManager import HistoricalDataManager
    from DataValidator import DataValidator
    
    # Initialize components
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    preprocessor = DataPreprocessor(debug=True)
    
    # Load and process data
    symbol = 'AAPL'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    # Load initial data
    price_data = hdm.get_price_data(symbol, start_date, end_date)
    raw_news_data = hdm.get_news_data(symbol, start_date, end_date)
    _, _, valid_news_data = validator.validate_news_data(raw_news_data)
    
    # Process all timeframes
    preprocessor.align_all_timeframes(price_data, valid_news_data)
    
    # Example: Access data for specific date
    test_date = "2024-01-15"
    print(f"\nGetting data for {test_date}")
    
    # Get daily data
    daily_df, news_data = preprocessor.get_data_for_date(test_date, '1d')
    if not daily_df.empty:
        print("\nDaily Data:")
        print(daily_df[['Date', 'Close', 'news_count']].to_string())
        
        # Get news for this day
        news_items = preprocessor.get_news_for_period(test_date, '1d')
        if news_items:
            print("\nNews Items:")
            for item in news_items:
                print(f"\nTime: {item['date']}")
                print(f"Title: {item['title']}")
    
    # Get hourly data for the same day
    hourly_df, _ = preprocessor.get_data_for_date(test_date, '1h')
    if not hourly_df.empty:
        print("\nHourly Data:")
        print(hourly_df[['Date', 'Close', 'news_count']].to_string())

if __name__ == "__main__":
    main()