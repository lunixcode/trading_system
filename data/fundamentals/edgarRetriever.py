import yfinance as yf
from datetime import datetime
import pandas as pd
from pathlib import Path
import os
import json
from typing import Dict
import logging

def setup_logging(log_path: Path):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / 'fundamentals_collector.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('FundamentalsCollector')

class FundamentalsCollector:
    def __init__(self, data_path: str):
        """Initialize fundamentals collector with data path"""
        self.data_path = Path(data_path)
        self.symbols = ['NVDA', 'META', 'GOOGL', 'AMZN', 'AAPL']
        
        # Setup logging
        log_path = self.data_path / 'logs'
        os.makedirs(log_path, exist_ok=True)
        self.logger = setup_logging(log_path)
        
        os.makedirs(self.data_path, exist_ok=True)

    def convert_to_serializable(self, obj):
        """Convert an object to JSON serializable format"""
        try:
            if isinstance(obj, list):
                return [self.convert_to_serializable(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='index')
            elif hasattr(obj, 'dtype'):
                return obj.item() if hasattr(obj, 'item') else float(obj)
            elif obj is None:
                return None
            elif not isinstance(obj, (str, int, float, bool, dict)):
                return str(obj)
            return obj
        except Exception as e:
            self.logger.error(f"Error converting object: {str(e)}")
            return str(obj)

    def process_dataframe(self, df):
        """Process dataframe with proper index handling"""
        try:
            if df is None or df.empty:
                return {}
            
            # Convert DataFrame to dictionary
            result = {}
            
            # Process each row
            for idx in df.index:
                # Convert index to string based on its type
                if isinstance(idx, pd.Timestamp):
                    key = idx.strftime('%Y-%m-%d')
                else:
                    key = str(idx)
                
                # Process the row data
                row_data = {}
                for col in df.columns:
                    val = df.loc[idx, col]
                    # Convert value based on type
                    if pd.isna(val):
                        row_data[str(col)] = None
                    elif isinstance(val, (int, float)):
                        row_data[str(col)] = float(val)
                    else:
                        row_data[str(col)] = str(val)
                        
                result[key] = row_data
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing DataFrame: {str(e)}")
            # Add more debug info
            self.logger.debug(f"DataFrame info: {df.info()}")
            self.logger.debug(f"Index type: {type(df.index)}")
            return {}

    

    def fetch_company_fundamentals(self, symbol: str) -> Dict:
        """Fetch latest company fundamentals"""
        self.logger.info(f"Fetching {symbol} fundamentals...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get fundamental data
            fundamentals = {
                'info': {
                    k: self.convert_to_serializable(v)
                    for k, v in ticker.info.items()
                },
                'income_statement': self.process_dataframe(ticker.income_stmt),
                'quarterly_income_statement': self.process_dataframe(ticker.quarterly_income_stmt),
                'balance_sheet': self.process_dataframe(ticker.balance_sheet),
                'quarterly_balance_sheet': self.process_dataframe(ticker.quarterly_balance_sheet),
                'cash_flow': self.process_dataframe(ticker.cash_flow),
                'quarterly_cash_flow': self.process_dataframe(ticker.quarterly_cashflow)
            }
            
            self.logger.info(f"Retrieved fundamental data for {symbol}")
            return fundamentals
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} fundamentals: {str(e)}")
            self.logger.exception("Full traceback:")
            return {}

    def save_fundamentals(self, fundamentals_data: Dict, symbol: str):
        """Save fundamentals data to JSON file"""
        filename = f"{symbol}_2024.json"
        file_path = self.data_path / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(fundamentals_data, f, indent=4)
            self.logger.info(f"Saved fundamentals data to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {str(e)}")
            # Test each section separately
            for key, value in fundamentals_data.items():
                try:
                    json.dumps({key: value})
                except Exception as e:
                    self.logger.error(f"Problem with section '{key}': {str(e)}")

    def collect_fundamentals(self):
        """Collect latest fundamentals for all symbols"""
        self.logger.info("\nStarting fundamentals collection...")
        
        for symbol in self.symbols:
            self.logger.info(f"\nProcessing {symbol}...")
            fundamentals = self.fetch_company_fundamentals(symbol)
            if fundamentals:
                self.save_fundamentals(fundamentals, symbol)
            else:
                self.logger.warning(f"Failed to collect data for {symbol}")

def main():
    # Configure paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = Path(current_dir).parent.parent
    fundamentals_path = project_root / 'data' / 'fundamentals'
    
    # Create collector and run
    collector = FundamentalsCollector(str(fundamentals_path))
    collector.collect_fundamentals()
    
    print("\nFundamentals collection complete!")

if __name__ == "__main__":
    main()