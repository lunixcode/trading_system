import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os
import json
from typing import List, Dict

class MT5Collector:
    def __init__(self, data_path: str):
        """
        Initialize MT5 connection and setup paths
        
        Args:
            data_path (str): Path to save bar data
        """
        self.data_path = data_path
        self.symbols = ['NVDA.NAS', 'MVRS.NAS', 'GOOG.NAS', 'AMZN.NAS', 'AAPL.NAS']
        
        # Initialize MT5 connection
        if not mt5.initialize():
            raise Exception(f"MT5 initialization failed: {mt5.last_error()}")
            
    def _get_month_range(self, year: int, month: int) -> tuple:
        """Get start and end datetime for a given month"""
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        return start_date, end_date

    def fetch_minute_bars(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        """
        Fetch 1-minute bars for a symbol and time period
        """
        start_date, end_date = self._get_month_range(year, month)
        
        # Fetch the bars
        bars = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M1,
            start_date,
            end_date
        )
        
        if bars is None:
            print(f"No data received for {symbol} in {year}-{month:02d}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df

    def save_bars(self, df: pd.DataFrame, symbol: str, year: int, month: int):
        """Save bar data to CSV file"""
        # Create directory structure
        month_dir = Path(self.data_path) / str(year) / f"{month:02d}"
        os.makedirs(month_dir, exist_ok=True)
        
        # Save file with format SYMBOL_YEAR_MONTH.csv
        filename = f"{symbol}_{year}_{month:02d}.csv"
        file_path = month_dir / filename
        
        df.to_csv(file_path, index=False)
        print(f"Saved {len(df)} bars to {file_path}")

    def collect_bars_range(self, start_month: int, start_year: int, end_month: int, end_year: int):
        """
        Collect bars for a range of months with confirmation between each month
        """
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)
        
        while current <= end:
            month_str = current.strftime('%B %Y')
            print(f"\n{'='*50}")
            print(f"Ready to collect bars for {month_str}")
            print(f"{'='*50}")
            
            # Ask for confirmation
            while True:
                response = input(f"\nCollect bars for {month_str}? (yes/skip/quit): ").lower().strip()
                if response in ['yes', 'skip', 'quit']:
                    break
                print("Please enter 'yes' to continue, 'skip' to skip this month, or 'quit' to exit")
            
            if response == 'quit':
                print("\nStopping data collection.")
                break
            elif response == 'skip':
                print(f"\nSkipping {month_str}")
            else:
                print(f"\nCollecting bars for {month_str}")
                
                for symbol in self.symbols:
                    print(f"\nProcessing {symbol}...")
                    df = self.fetch_minute_bars(symbol, current.year, current.month)
                    if df is not None:
                        self.save_bars(df, symbol, current.year, current.month)
                
                input("\nPress Enter to continue to next month...")
            
            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

        print("\nBar collection complete!")
        
    def __del__(self):
        """Cleanup MT5 connection"""
        mt5.shutdown()

def main():
    # Configure paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = Path(current_dir).parent.parent
    
    # Point to data/bars directory
    bars_path = project_root / 'data' / 'price' / 'raw'
    
    # Create collector and run
    collector = MT5Collector(str(bars_path))
    
    # Collect from January 2024 to current month
    current_date = datetime.now()
    collector.collect_bars_range(1, 2024, current_date.month, current_date.year)

if __name__ == "__main__":
    main()