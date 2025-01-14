from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import json

# Type hints imports
if TYPE_CHECKING:
    from backtesting.framework.DataPreprocessor import DataPreprocessor

class EventDataExtractor:
    """
    Extracts and saves 2-week periods of aligned data around significant market events.
    """
    
    def __init__(self, 
                 aligned_dir: str = "data/aligned/2week",
                 lookback_days: int = 7,
                 lookforward_days: int = 7,
                 debug: bool = False):
        """Initialize the EventDataExtractor."""
        self.debug = debug
        self.aligned_dir = Path(aligned_dir)
        self.lookback_days = lookback_days
        self.lookforward_days = lookforward_days
        
        # Create aligned data directory if it doesn't exist
        self.aligned_dir.mkdir(parents=True, exist_ok=True)
        
        if self.debug:
            print(f"Initialized EventDataExtractor")
            print(f"Aligned data directory: {self.aligned_dir}")
    
    def _get_aligned_paths(self, symbol: str, event_number: int) -> Tuple[Path, Path]:
        """Get the paths for saving aligned data and metadata."""
        # Create 5min and metadata directories
        data_dir = self.aligned_dir / symbol / "5min"
        meta_dir = data_dir / "metadata"
        
        # Create directories if they don't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Return paths for data and metadata files
        data_path = data_dir / f"{event_number}.parquet"
        meta_path = meta_dir / f"{event_number}.json"
        
        return data_path, meta_path
    
    def extract_event_data(self,
                          event_date: datetime,
                          symbol: str,
                          preprocessor: 'DataPreprocessor') -> Tuple[pd.DataFrame, Dict]:
        """
        Extract 2-week period data around an event.
        
        Args:
            event_date: Date of the event
            symbol: Stock symbol
            preprocessor: DataPreprocessor instance with aligned data
            
        Returns:
            Tuple of (event_period_data, event_details)
        """
        if self.debug:
            print(f"\nExtracting event data for {symbol} on {event_date}")
        
        # Calculate date range
        start_date = event_date - timedelta(days=self.lookback_days)
        end_date = event_date + timedelta(days=self.lookforward_days)
        
        if self.debug:
            print(f"Date range: {start_date} to {end_date}")
        
        # Get 5-minute data for the period
        if '5min' not in preprocessor.aligned_data:
            raise ValueError("5-minute timeframe data not available")
            
        df = preprocessor.aligned_data['5min'].copy()
        
        # Convert dates if needed
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter to event period
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        period_data = df[mask].copy()
        
        if period_data.empty:
            raise ValueError(f"No data found for period {start_date} to {end_date}")
        
        # Create event details
        event_details = {
            'symbol': symbol,
            'event_date': event_date.isoformat(),
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'data_points': len(period_data),
            'timeframe': '5min'
        }
        
        return period_data, event_details
    
    def save_event_data(self,
                       period_data: pd.DataFrame,
                       event_details: Dict,
                       symbol: str,
                       event_number: int) -> Path:
        """Save event period data and metadata."""
        try:
            # Get save paths
            data_file, meta_file = self._get_aligned_paths(symbol, event_number)
            
            # Save aligned data
            period_data.to_parquet(data_file)
            
            # Save metadata
            with open(meta_file, 'w') as f:
                json.dump(event_details, f, indent=2, default=str)
            
            if self.debug:
                print(f"Event {event_number} data saved to: {data_file}")
                print(f"Metadata saved to: {meta_file}")
            
            return data_file.parent  # Return the 5min directory
            
        except Exception as e:
            if self.debug:
                print(f"Error saving event data: {str(e)}")
            raise

    def extract_and_save_event(self,
                             event_date: datetime,
                             symbol: str,
                             preprocessor: 'DataPreprocessor',
                             event_number: int) -> Path:
        """Extract and save event data in one operation."""
        # Extract data
        period_data, event_details = self.extract_event_data(
            event_date,
            symbol,
            preprocessor
        )
        
        # Add event number to details
        event_details['event_number'] = event_number
        
        # Save data with numbered files
        save_path = self.save_event_data(
            period_data,
            event_details,
            symbol,
            event_number
        )
        
        return save_path