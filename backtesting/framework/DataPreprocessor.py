from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataPreprocessor:
    """
    Handles the preprocessing of financial data including alignment of different data sources,
    feature engineering, and data cleaning.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.timeframes = {
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1h': 'H',
            '1d': 'D'
        }

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
        
        # Extract date components based on timeframe
        for df in [news_df, resampled_price]:
            df['year'] = df['date' if 'date' in df else 'Date'].dt.year
            df['month'] = df['date' if 'date' in df else 'Date'].dt.month
            df['day'] = df['date' if 'date' in df else 'Date'].dt.day
            df['hour'] = df['date' if 'date' in df else 'Date'].dt.hour
            df['minute'] = df['date' if 'date' in df else 'Date'].dt.minute
        
        # Add news index
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

    def align_all_timeframes(self, price_data: pd.DataFrame, news_data: List[dict]) -> Dict[str, pd.DataFrame]:
        """
        Align data for all timeframes.
        
        Returns:
            Dictionary with aligned data for each timeframe
        """
        if self.debug:
            print("\nProcessing all timeframes...")
        
        aligned_data = {}
        
        for timeframe in self.timeframes.keys():
            if self.debug:
                print(f"\nProcessing {timeframe} timeframe...")
            
            aligned_df, news = self.align_price_and_news(price_data, news_data, timeframe)
            aligned_data[timeframe] = aligned_df
        
        return aligned_data, news

def main():
    """Test function for the DataPreprocessor"""
    from HistoricalDataManager import HistoricalDataManager
    from DataValidator import DataValidator
    
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    preprocessor = DataPreprocessor(debug=True)
    
    symbol = 'AAPL'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    try:
        # Load and validate data
        price_data = hdm.get_price_data(symbol, start_date, end_date)
        raw_news_data = hdm.get_news_data(symbol, start_date, end_date)
        _, _, valid_news_data = validator.validate_news_data(raw_news_data)
        
        # Process all timeframes
        aligned_data_dict, news_data = preprocessor.align_all_timeframes(price_data, valid_news_data)
        
        # Save aligned data for each timeframe
        for timeframe, aligned_data in aligned_data_dict.items():
            output_file = f"{symbol}_aligned_{timeframe}.csv"
            aligned_data.to_csv(output_file, index=False)
            print(f"\nSaved {timeframe} aligned data to {output_file}")
            print(f"Shape: {aligned_data.shape}")
            print(f"Periods with news: {(aligned_data['news_count'] > 0).sum()}")
        
        # Save news data
        news_file = f"{symbol}_news_data.json"
        import json
        with open(news_file, 'w') as f:
            json.dump(news_data, f, indent=2)
        print(f"\nSaved news data to {news_file}")
        
        # Show sample for each timeframe
        print("\nSample of aligned data for each timeframe:")
        for timeframe, aligned_data in aligned_data_dict.items():
            news_periods = aligned_data[aligned_data['news_count'] > 0]
            if not news_periods.empty:
                print(f"\n{timeframe} Timeframe Sample:")
                sample = news_periods[['Date', 'news_count', 'Close', 'news_indices']].head(3)
                print(sample.to_string())
                
                # Show sample news for first period
                first_period = sample.iloc[0]
                print(f"\nSample news for {timeframe} period {first_period['Date']}:")
                indices = [int(i) for i in first_period['news_indices'].split(',')]
                for news_idx in indices[:2]:  # Show first 2 news items
                    print(f"News [{news_idx}]: {news_data[news_idx]['title']}")
        
    except Exception as e:
        import traceback
        print(f"Error during processing: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()