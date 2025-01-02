from datetime import datetime, timedelta
import requests
import sqlite3
import time
import os
import json
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

class NewsCollector:
    def __init__(self, db_path: str, raw_data_path: str):
        """
        Initialize NewsCollector with database and raw data paths
        
        Args:
            db_path (str): Path to SQLite database
            raw_data_path (str): Path to store raw JSON files
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv('EOD_API_KEY')
        if not self.api_key:
            raise ValueError("EOD_API_KEY not found in .env file")
            
        self.db_path = db_path
        self.raw_data_path = raw_data_path
        self.base_url = "https://eodhistoricaldata.com/api/news"
        self.symbols = ['NVDA', 'META', 'GOOGL', 'AMZN', 'AAPL']
        
        # Create necessary directories
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._setup_database()

    def _setup_database(self):
        """Create the news table if it doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    published_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    source TEXT,
                    symbols TEXT,
                    sentiment REAL,
                    importance INTEGER,
                    raw_data TEXT,
                    json_file_path TEXT
                )
            """)

    def _get_month_range(self, year: int, month: int) -> tuple:
        """
        Get start and end dates for a given month
        
        Args:
            year (int): Year
            month (int): Month (1-12)
            
        Returns:
            tuple: (start_date, end_date) in format 'YYYY-MM-DD'
        """
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    def _save_raw_json(self, data: List[Dict], symbol: str, year: int, month: int) -> str:
        """
        Save raw JSON data to file with format SYMBOL_YEAR_MONTH
        """
        # Create year/month subdirectories
        month_dir = Path(self.raw_data_path) / str(year) / f"{month:02d}"
        os.makedirs(month_dir, exist_ok=True)
        
        # Create simple filename
        filename = f"{symbol}_{year}_{month:02d}.json"
        
        file_path = month_dir / filename
        
        # Save JSON data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        return str(file_path)
    def fetch_news_for_month(self, year: int, month: int) -> List[Dict]:
        """
        Fetch news for all symbols for a specific month
        
        Args:
            year (int): Year to fetch news for
            month (int): Month to fetch news for (1-12)
            
        Returns:
            List[Dict]: List of news articles with file paths
        """
        start_date, end_date = self._get_month_range(year, month)
        all_news = []
        
        for symbol in self.symbols:
            page = 0
            while True:  # Keep fetching until no more results
                params = {
                    'api_token': self.api_key,
                    's': symbol,  # Changed from 'symbol' to 's' for specific stock news
                    'from': start_date,
                    'to': end_date,
                    #'limit': 10,  # Test limit of 10 articles per company
                    'limit': 1000,  # Maximum allowed per request - uncomment for full collection
                    'offset': page * 10,  # Adjusted offset to match new limit
                    # 'offset': page * 1000,  # Uncomment for full collection
                    'filter': 'all'  # Get all news specifically mentioning this stock
                }
                
                try:
                    print(f"Fetching page {page + 1} for {symbol} ({start_date} to {end_date})")
                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()
                    news_data = response.json()
                    
                    # Print the API URL for debugging (without API key)
                    debug_params = params.copy()
                    debug_params['api_token'] = 'HIDDEN'
                    #print(f"API URL: {self.base_url}?{requests.utils.urlencode(debug_params)}")
                    
                    # If no more news articles, break the loop
                    if not news_data:
                        print(f"No more news for {symbol}")
                        break
                    
                    # Save raw JSON and get file path
                    json_file_path = self._save_raw_json(news_data, symbol, year, month)
                    
                    # Add symbol and file path to each news item
                    for item in news_data:
                        item['queried_symbol'] = symbol
                        item['json_file_path'] = json_file_path
                    
                    all_news.extend(news_data)
                    
                    print(f"Retrieved {len(news_data)} articles for {symbol} - {year}-{month:02d} (page {page + 1})")
                    
                    # For testing: break after first page to limit to 10 articles
                    break
                    
                    # Original pagination logic - uncomment for full collection
                    # # If we got less than the limit, there are no more pages
                    # if len(news_data) < 1000:
                    #     break
                        
                    page += 1
                    
                    # Rate limiting - 1 request per second
                    time.sleep(1)
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching news for {symbol}: {e}")
                    break
                
                except Exception as e:
                    print(f"Unexpected error for {symbol}: {str(e)}")
                    print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
                    break
        
        unique_news = self.deduplicate_news(all_news)
        print(f"After deduplication: {len(unique_news)} unique articles from {len(all_news)} total")
        return unique_news
    

    def deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:  # Removed underscore
        """
        Remove duplicate news articles based on title and published date
        
        Args:
            news_list (List[Dict]): List of news articles
            
        Returns:
            List[Dict]: Deduplicated list of news articles
        """
        seen = set()
        unique_news = []
        
        for news in news_list:
            key = (news.get('title'), news.get('published_at'))
            if key not in seen:
                seen.add(key)
                unique_news.append(news)
        
        return unique_news
    

    def save_to_database(self, news_articles: List[Dict]):
        """
        Save news articles to SQLite database
        
        Args:
            news_articles (List[Dict]): List of news articles to save
        """
        with sqlite3.connect(self.db_path) as conn:
            for article in news_articles:
                try:
                    # Convert symbols to string if it's a list or other type
                    symbols = article.get('symbols', '')
                    if isinstance(symbols, (list, tuple)):
                        symbols = ','.join(str(s) for s in symbols)
                    elif not isinstance(symbols, str):
                        symbols = str(symbols)

                    # Print debugging info for failed articles
                    print(f"Processing article: {article.get('title', '')}")
                    print(f"Symbols type: {type(symbols)}, value: {symbols}")
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO news 
                        (id, title, content, published_at, updated_at, source, symbols, raw_data, json_file_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        article.get('id', ''),
                        article.get('title', ''),
                        article.get('content', ''),
                        article.get('published_at', ''),
                        article.get('updated_at', ''),
                        article.get('source', ''),
                        symbols,  # Now properly formatted as string
                        json.dumps(article),  # Properly serialize the raw data
                        article.get('json_file_path', '')
                    ))
                except sqlite3.Error as e:
                    print(f"Error saving article: {article.get('title', 'Unknown Title')}")
                    print(f"Error details: {str(e)}")
                    print(f"Article data: {json.dumps(article, indent=2)}")
                    continue

    def collect_news_range(self, start_month: int, start_year: int, end_month: int, end_year: int):
        """
        Collect news for a range of months with confirmation between each month
        
        Args:
            start_month (int): Starting month (1-12)
            start_year (int): Starting year
            end_month (int): Ending month (1-12)
            end_year (int): Ending year
        """
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)
        
        while current <= end:
            month_str = current.strftime('%B %Y')
            print(f"\n{'='*50}")
            print(f"Ready to collect news for {month_str}")
            print(f"{'='*50}")
            
            # Ask for confirmation
            while True:
                response = input(f"\nCollect news for {month_str}? (yes/skip/quit): ").lower().strip()
                if response in ['yes', 'skip', 'quit']:
                    break
                print("Please enter 'yes' to continue, 'skip' to skip this month, or 'quit' to exit")
            
            if response == 'quit':
                print("\nStopping news collection.")
                break
            elif response == 'skip':
                print(f"\nSkipping {month_str}")
            else:
                print(f"\nCollecting news for {month_str}")
                news = self.fetch_news_for_month(current.year, current.month)
                self.save_to_database(news)
                print(f"\nSaved {len(news)} articles for {month_str}")
                
                # Show directory where files were saved
                month_dir = Path(self.raw_data_path) / str(current.year) / f"{current.month:02d}"
                print(f"\nRaw JSON files saved in: {month_dir}")
                
                # Optional: Show some stats about the data collected
                print("\nCollection Summary:")
                print(f"- Total articles: {len(news)}")
                for symbol in self.symbols:
                    symbol_articles = [n for n in news if n.get('queried_symbol') == symbol]
                    print(f"- {symbol}: {len(symbol_articles)} articles")
                
                input("\nPress Enter to continue to next month...")
            
            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        print("\nNews collection complete!")

def main():
     # Configure paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = Path(current_dir).parent.parent  # Go up two levels from realtime/news to project root
    
    # Point directly to data/news directories
    db_path = project_root / 'data' / 'news' / 'db' / 'news.db'
    raw_data_path = project_root / 'data' / 'news' / 'raw'
    
    # Find and load .env file
    env_path = project_root / '.env'
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    # Create news collector instance
    collector = NewsCollector(str(db_path), str(raw_data_path))
    
    # Collect news from January 2024 to current month
    current_date = datetime.now()
    collector.collect_news_range(1, 2023, 1, 2024)

if __name__ == "__main__":
    main()