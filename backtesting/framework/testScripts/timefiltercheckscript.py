import os
import sys
# Set sys.path to include the 'framework' directory explicitly
framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(framework_path)

from datetime import datetime
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator

def main():
    # Initialize with debug mode
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    
    # Test parameters
    symbol = 'AAPL'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    print("\nTesting news data validation and filtering...")
    try:
        # Load news data
        news_data = hdm.get_news_data(symbol, start_date, end_date)
        print(f"\nLoaded {len(news_data)} total news items")
        
        # Validate and filter news data
        is_valid, issues, filtered_news = validator.validate_news_data(news_data)
        
        print("\nProcessing Results:")
        print(f"Original news items: {len(news_data)}")
        print(f"Filtered news items: {len(filtered_news)}")
        print(f"Items removed: {len(news_data) - len(filtered_news)}")
        
        if filtered_news:
            print("\nSample of valid news items (first 3):")
            for idx, item in enumerate(filtered_news[:3]):
                print(f"\nItem {idx + 1}:")
                print(f"Date: {item['date']}")
                print(f"Title: {item['title'][:100]}...")  # Show first 100 chars of title
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()