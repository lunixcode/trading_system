import os
import sys
# Set sys.path to include the 'framework' directory explicitly
framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(framework_path)
from datetime import datetime
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator
from DataPreprocessor import DataPreprocessor

# Initialize components
hdm = HistoricalDataManager()
validator = DataValidator()
preprocessor = DataPreprocessor()

# Set date range (just for one day in this case)
date = "2024-01-15"
symbol = "AAPL"
start_date = datetime(2024, 1, 1)  # Include some buffer to ensure we have the data
end_date = datetime(2024, 3, 31)

# Load data
price_data = hdm.get_price_data(symbol, start_date, end_date)
raw_news_data = hdm.get_news_data(symbol, start_date, end_date)

# Validate news
_, _, valid_news_data = validator.validate_news_data(raw_news_data)

# Process data
aligned_data, news_data = preprocessor.align_all_timeframes(price_data, valid_news_data)

# Get data for specific date
daily_data, _ = preprocessor.get_data_for_date(date, '1d')

# Print results
if not daily_data.empty:
    print(f"\nNews for {date}:")
    news_items = preprocessor.get_news_for_period(date, '1d')
    for item in news_items:
        print(f"\nTime: {item['date']}")
        print(f"Title: {item['title']}")