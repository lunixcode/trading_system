# datapullTest.py
from datetime import datetime
import pandas as pd
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator
from DataPreprocessor import DataPreprocessor

class DataAccessor:
    def __init__(self, symbol: str, start_date: datetime, end_date: datetime, debug: bool = False):
        """Initialize the data pipeline and load data."""
        self.hdm = HistoricalDataManager(debug=debug)
        self.validator = DataValidator(debug=debug)
        self.preprocessor = DataPreprocessor(debug=debug)
        
        # Load and process data
        print(f"\nLoading data for {symbol}...")
        price_data = self.hdm.get_price_data(symbol, start_date, end_date)
        raw_news_data = self.hdm.get_news_data(symbol, start_date, end_date)
        _, _, valid_news_data = self.validator.validate_news_data(raw_news_data)
        
        # Process all timeframes
        print("Processing timeframes...")
        self.aligned_data, self.news_data = self.preprocessor.align_all_timeframes(
            price_data, 
            valid_news_data
        )
        print("Data loading complete!")

    def get_date_data(self, date: str, timeframe: str = '1d'):
        """Get price and news data for a specific date."""
        data, _ = self.preprocessor.get_data_for_date(date, timeframe)
        if data.empty:
            print(f"No data found for {date}")
            return
            
        print(f"\nData for {date} ({timeframe} timeframe):")
        print("\nPrice Data:")
        print(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'news_count']].to_string())
        
        if data['news_count'].sum() > 0:
            print("\nNews Items:")
            news_items = self.preprocessor.get_news_for_period(date, timeframe)
            for item in news_items:
                print(f"\nTime: {item['date']}")
                print(f"Title: {item['title']}")
                print(f"Link: {item['link']}")

    def get_news_between_dates(self, start_date: str, end_date: str, timeframe: str = '1d'):
        """Get all news between two dates."""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        df = self.preprocessor.aligned_data[timeframe]
        mask = (df['Date'] >= start_dt) & (df['Date'] <= end_dt)
        period_data = df[mask]
        
        if period_data.empty:
            print(f"No data found between {start_date} and {end_date}")
            return
            
        print(f"\nNews between {start_date} and {end_date}:")
        for _, row in period_data[period_data['news_count'] > 0].iterrows():
            print(f"\nDate: {row['Date']}")
            if row['news_indices']:  # Check if there are any news indices
                indices = [int(idx) for idx in row['news_indices'].split(',')]
                for idx in indices:
                    news = self.news_data[idx]
                    print(f"- [{news['date']}] {news['title']}")

    def print_available_dates(self, timeframe: str = '1d'):
        """Show available dates in the dataset."""
        df = self.preprocessor.aligned_data[timeframe]
        print(f"\nDate Range for {timeframe} timeframe:")
        print(f"Start: {df['Date'].min()}")
        print(f"End: {df['Date'].max()}")
        print(f"Total periods: {len(df)}")
        print(f"Periods with news: {(df['news_count'] > 0).sum()}")

def main():
    # Initialize with your data
    symbol = 'AAPL'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    # Create accessor
    accessor = DataAccessor(symbol, start_date, end_date, debug=True)
    
    while True:
        print("\nData Access Options:")
        print("1. View specific date")
        print("2. View date range")
        print("3. Show available dates")
        print("4. Change timeframe")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            date = input("Enter date (YYYY-MM-DD): ")
            timeframe = input("Enter timeframe (5min/15min/30min/1h/1d) [default: 1d]: ") or '1d'
            accessor.get_date_data(date, timeframe)
            
        elif choice == '2':
            start = input("Enter start date (YYYY-MM-DD): ")
            end = input("Enter end date (YYYY-MM-DD): ")
            timeframe = input("Enter timeframe (5min/15min/30min/1h/1d) [default: 1d]: ") or '1d'
            accessor.get_news_between_dates(start, end, timeframe)
            
        elif choice == '3':
            timeframe = input("Enter timeframe (5min/15min/30min/1h/1d) [default: 1d]: ") or '1d'
            accessor.print_available_dates(timeframe)
            
        elif choice == '4':
            print("\nAvailable timeframes:")
            print("- 5min  (5-minute data)")
            print("- 15min (15-minute data)")
            print("- 30min (30-minute data)")
            print("- 1h    (1-hour data)")
            print("- 1d    (Daily data)")
            
        elif choice == '5':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()