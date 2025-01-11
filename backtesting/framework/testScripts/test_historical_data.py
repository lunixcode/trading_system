# test_news_time.py
import os
import sys
# Set sys.path to include the 'framework' directory explicitly
framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(framework_path)
from datetime import datetime
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator

def main():
    # Initialize both classes with debug mode
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    
    # Test parameters
    symbol = 'AAPL'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    print("\nTesting news data time validation...")
    try:
        # Load news data
        news_data = hdm.get_news_data(symbol, start_date, end_date)
        print(f"\nLoaded {len(news_data)} news items")
        
        # Count and display items with zero times
        zero_times = []
        non_zero_times = []
        
        for idx, item in enumerate(news_data):
            news_date = datetime.fromisoformat(item['date'])
            if news_date.hour == 0 and news_date.minute == 0 and news_date.second == 0:
                zero_times.append((idx, item['date']))
            else:
                non_zero_times.append((idx, item['date']))
        
        print(f"\nTime Analysis:")
        print(f"Total items: {len(news_data)}")
        print(f"Items with zero times (00:00:00): {len(zero_times)}")
        print(f"Items with valid times: {len(non_zero_times)}")
        
        # Show some examples
        if zero_times:
            print("\nExample items with zero times:")
            for idx, date in zero_times[:5]:  # Show first 5
                print(f"Index {idx}: {date}")
        
        if non_zero_times:
            print("\nExample items with valid times:")
            for idx, date in non_zero_times[:5]:  # Show first 5
                print(f"Index {idx}: {date}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()