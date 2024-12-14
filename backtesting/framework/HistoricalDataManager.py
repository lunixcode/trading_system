import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

class HistoricalDataManager:
    def __init__(self, base_path: str = "data", debug: bool = False):
        self.debug = debug
        if self.debug:
            print(f"Initializing HistoricalDataManager with base path: {base_path}")
            
        self.base_path = Path(base_path)
        self.price_path = self.base_path / "price" / "raw"
        self.news_path = self.base_path / "news" / "raw"
        self.fundamentals_path = self.base_path / "fundamentals"
        
        if self.debug:
            print(f"Checking paths:")
            print(f"Price path exists: {self.price_path.exists()}")
            print(f"News path exists: {self.news_path.exists()}")
            print(f"Fundamentals path exists: {self.fundamentals_path.exists()}")
        
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._news_cache: Dict[str, List[dict]] = {}
        self._fundamentals_cache: Dict[str, dict] = {}

    def get_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load price data for a given symbol and date range."""
        if self.debug:
            print(f"Loading price data for {symbol} from {start_date} to {end_date}")
        
        cache_key = f"{symbol}_{start_date.year}"
        if self.debug:
            print(f"Cache key: {cache_key}")
        
        if cache_key not in self._price_cache:
            if self.debug:
                print(f"Data not in cache, loading from files...")
            dfs = []
            current_date = start_date
            
            while current_date <= end_date:
                file_path = self.price_path / str(current_date.year) / f"{current_date.month:02d}" / f"{symbol}.NAS_{current_date.year}_{current_date.month:02d}.csv"
                if self.debug:
                    print(f"Checking file: {file_path}")
                
                if file_path.exists():
                    if self.debug:
                        print(f"Loading file: {file_path}")
                    try:
                        df = pd.read_csv(file_path)
                        column_mapping = {
                            'time': 'Date',
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'tick_volume': 'Tick_Volume',
                            'spread': 'Spread',
                            'real_volume': 'Volume'
                        }
                        df = df.rename(columns=column_mapping)
                        df['Date'] = pd.to_datetime(df['Date'])
                        
                        if self.debug:
                            print(f"Loaded {len(df)} rows from {file_path}")
                            print(f"Columns in dataframe: {df.columns.tolist()}")
                        dfs.append(df)
                    except Exception as e:
                        if self.debug:
                            print(f"Error loading file {file_path}: {str(e)}")
                        continue
                else:
                    if self.debug:
                        print(f"File not found: {file_path}")
                
                current_date = current_date.replace(month=current_date.month % 12 + 1)
                if current_date.month == 1:
                    current_date = current_date.replace(year=current_date.year + 1)
            
            if not dfs:
                raise FileNotFoundError(f"No price data found for {symbol} in specified date range")
            
            self._price_cache[cache_key] = pd.concat(dfs, ignore_index=True)
            if self.debug:
                print(f"Cached {len(self._price_cache[cache_key])} rows of price data for {cache_key}")
        else:
            if self.debug:
                print(f"Using cached data for {cache_key}")
        
        mask = (self._price_cache[cache_key]['Date'] >= start_date) & (self._price_cache[cache_key]['Date'] <= end_date)
        result = self._price_cache[cache_key][mask].copy()
        if self.debug:
            print(f"Returning {len(result)} rows of price data")
        return result

    def get_news_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[dict]:
        """Load news data for a given symbol and date range."""
        if self.debug:
            print(f"Loading news data for {symbol} from {start_date} to {end_date}")
        
        news_items = []
        current_date = start_date
        
        while current_date <= end_date:
            file_path = self.news_path / str(current_date.year) / f"{current_date.month:02d}" / f"{symbol}_{current_date.year}_{current_date.month:02d}.json"
            if self.debug:
                print(f"Checking news file: {file_path}")
            
            try:
                if file_path.exists():
                    if self.debug:
                        print(f"Loading news file: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        month_news = json.load(f)
                        if self.debug:
                            print(f"Loaded {len(month_news)} news items from {file_path}")
                        
                        # Filter news items within the date range
                        for item in month_news:
                            try:
                                news_date = datetime.fromisoformat(item['date'])
                                # Convert to naive datetime if timezone aware
                                if news_date.tzinfo is not None:
                                    news_date = news_date.replace(tzinfo=None)
                                
                                # Convert input dates to naive if they're timezone aware
                                start_compare = start_date if start_date.tzinfo is None else start_date.replace(tzinfo=None)
                                end_compare = end_date if end_date.tzinfo is None else end_date.replace(tzinfo=None)
                                
                                if start_compare <= news_date <= end_compare:
                                    news_items.append(item)
                            except ValueError as ve:
                                if self.debug:
                                    print(f"Error parsing date in news item: {str(ve)}")
                                continue
                            except Exception as e:
                                if self.debug:
                                    print(f"Error processing news item: {str(e)}")
                                continue
                else:
                    if self.debug:
                        print(f"News file not found: {file_path}")
            except Exception as e:
                if self.debug:
                    print(f"Error processing news file {file_path}: {str(e)}")
                continue
            
            current_date = current_date.replace(month=current_date.month % 12 + 1)
            if current_date.month == 1:
                current_date = current_date.replace(year=current_date.year + 1)
        
        if self.debug:
            print(f"Returning {len(news_items)} news items")
            if news_items:
                print("Sample of first news item date:", news_items[0]['date'])
        
        return news_items

    def clear_cache(self, data_type: Optional[str] = None):
        """Clear the data cache."""
        if self.debug:
            print(f"Clearing cache for data_type: {data_type if data_type else 'all'}")
        
        if data_type is None:
            self._price_cache.clear()
            self._news_cache.clear()
            self._fundamentals_cache.clear()
            if self.debug:
                print("Cleared all caches")
        elif data_type == 'price':
            self._price_cache.clear()
            if self.debug:
                print("Cleared price cache")
        elif data_type == 'news':
            self._news_cache.clear()
            if self.debug:
                print("Cleared news cache")
        elif data_type == 'fundamentals':
            self._fundamentals_cache.clear()
            if self.debug:
                print("Cleared fundamentals cache")
        else:
            raise ValueError("Invalid data_type. Must be 'price', 'news', 'fundamentals', or None")