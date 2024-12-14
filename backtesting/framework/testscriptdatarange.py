from datetime import datetime
from HistoricalDataManager import HistoricalDataManager

def main():
    # Initialize with debug mode
    hdm = HistoricalDataManager(debug=True)
    
    # Test parameters
    symbol = 'AAPL'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    print(f"\nChecking price data for {symbol}")
    try:
        # Load price data
        price_data = hdm.get_price_data(symbol, start_date, end_date)
        
        # Print date range info
        print("\nPrice Data Range:")
        print(f"First date in data: {price_data['Date'].min()}")
        print(f"Last date in data: {price_data['Date'].max()}")
        print(f"Total trading days: {len(price_data)}")
        
        # Print first few and last few dates to verify continuity
        print("\nFirst 5 trading days:")
        print(price_data['Date'].head().tolist())
        
        print("\nLast 5 trading days:")
        print(price_data['Date'].tail().tolist())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()