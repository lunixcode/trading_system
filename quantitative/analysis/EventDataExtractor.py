from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

# Type hints imports
if TYPE_CHECKING:
    from backtesting.framework.DataPreprocessor import DataPreprocessor
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import json

class EventDataExtractor:
    """
    Extracts and saves 2-week periods of aligned data around significant market events.
    """
    
    def __init__(self, 
                 aligned_dir: str = "data/aligned/2week",
                 lookback_days: int = 7,
                 lookforward_days: int = 7,
                 debug: bool = False):
        """
        Initialize the EventDataExtractor.
        
        Args:
            aligned_dir: Directory for saving aligned data
            lookback_days: Days to look back before event
            lookforward_days: Days to look forward after event
            debug: Enable debug logging
        """
        self.debug = debug
        self.aligned_dir = Path(aligned_dir)
        self.lookback_days = lookback_days
        self.lookforward_days = lookforward_days
        
        # Create aligned data directory if it doesn't exist
        self.aligned_dir.mkdir(parents=True, exist_ok=True)
        
        if self.debug:
            print(f"Initialized EventDataExtractor")
            print(f"Aligned data directory: {self.aligned_dir}")
    
    def _get_aligned_path(self, symbol: str, event_date: datetime) -> Path:
        """Get the directory path for saving aligned event data."""
        date_str = event_date.strftime('%Y%m%d')
        return self.aligned_dir / symbol / date_str
    
    def extract_event_data(self,
                          event_date: datetime,
                          symbol: str,
                          preprocessor: 'DataPreprocessor'): # type: ignore) -> Tuple[pd.DataFrame, Dict]:
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
                       event_date: datetime) -> Path:
        """
        Save event period data to aligned data directory.
        
        Args:
            period_data: DataFrame containing the event period data
            event_details: Dictionary with event metadata
            symbol: Stock symbol
            event_date: Event date
            
        Returns:
            Path to saved data directory
        """
        # Get save location
        save_dir = self._get_aligned_path(symbol, event_date)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save aligned data
            data_file = save_dir / 'aligned_5min.parquet'
            period_data.to_parquet(data_file)
            
            # Save metadata
            meta_file = save_dir / 'metadata.json'
            with open(meta_file, 'w') as f:
                json.dump(event_details, f, indent=2, default=str)
            
            if self.debug:
                print(f"\nSaved event data:")
                print(f"Data file: {data_file}")
                print(f"Metadata: {meta_file}")
                print(f"Records: {len(period_data)}")
            
            return save_dir
            
        except Exception as e:
            if self.debug:
                print(f"Error saving event data: {str(e)}")
            raise
    
    def extract_and_save_event(self,
                             event_date: datetime,
                             symbol: str,
                             preprocessor: 'DataPreprocessor') -> Path:
        """
        Extract and save event data in one operation.
        
        Args:
            event_date: Date of the event
            symbol: Stock symbol
            preprocessor: DataPreprocessor instance
            
        Returns:
            Path to saved data directory
        """
        # Extract data
        period_data, event_details = self.extract_event_data(
            event_date,
            symbol,
            preprocessor
        )
        
        # Save data
        return self.save_event_data(
            period_data,
            event_details,
            symbol,
            event_date
        )
    
    def extract_multiple_events(self,
                              events: List[datetime],
                              symbol: str,
                              preprocessor: 'DataPreprocessor') -> List[Path]:
        """
        Extract and save data for multiple events.
        
        Args:
            events: List of event dates
            symbol: Stock symbol
            preprocessor: DataPreprocessor instance
            
        Returns:
            List of paths to saved data directories
        """
        saved_paths = []
        
        for event_date in events:
            try:
                save_path = self.extract_and_save_event(
                    event_date,
                    symbol,
                    preprocessor
                )
                saved_paths.append(save_path)
                
            except Exception as e:
                if self.debug:
                    print(f"Error processing event {event_date}: {str(e)}")
                continue
        
        return saved_paths