from datetime import datetime
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator
from DataPreprocessor import DataPreprocessor
import pandas as pd
import numpy as np

def main():
    print("\n=== Starting Data Preprocessing Pipeline ===")
    
    # Initialize all components with debug mode
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    preprocessor = DataPreprocessor(debug=True)
    
    # Test parameters
    symbol = 'AAPL'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    try:
        print("\n1. Loading Data...")
        # Load price data
        price_data = hdm.get_price_data(symbol, start_date, end_date)
        print(f"Loaded {len(price_data)} price records")
        
        # Load and validate news data
        raw_news_data = hdm.get_news_data(symbol, start_date, end_date)
        print(f"Loaded {len(raw_news_data)} news items")
        
        print("\n2. Validating News Data...")
        # Get only valid news items (with non-zero times)
        _, _, valid_news_data = validator.validate_news_data(raw_news_data)
        print(f"Valid news items: {len(valid_news_data)}")
        
        print("\n3. Preprocessing Data...")
        # Align price and news data
        aligned_data = preprocessor.align_price_and_news(price_data, valid_news_data)
        
        print("\n4. Summary Statistics:")
        print(f"Total rows in aligned data: {len(aligned_data)}")
        print(f"Rows with news: {(aligned_data['news_count'] > 0).sum()}")
        
        print("\n5. Sample of Data with News:")
        news_samples = aligned_data[aligned_data['news_count'] > 0].head()
        if not news_samples.empty:
            for idx, row in news_samples.iterrows():
                print(f"\nDate: {row['Date']}")
                print(f"News Count: {row['news_count']}")
                print(f"Price: {row['Close']}")
                print(f"News Titles: {row['news_titles'][:100]}...")  # First 100 chars
        else:
            print("No rows found with news")
        
        # Save sample to CSV for inspection
        sample_file = f"{symbol}_preprocessed_sample.csv"
        aligned_data.head(100).to_csv(sample_file)
        print(f"\nSaved first 100 rows to {sample_file}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()