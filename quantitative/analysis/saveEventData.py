#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# Import components
from backtesting.framework.HistoricalDataManager import HistoricalDataManager
from backtesting.framework.DataValidator import DataValidator
from backtesting.framework.DataPreprocessor import DataPreprocessor
from quantitative.analysis.StatisticalAnalyzer import StatisticalAnalyzer
from quantitative.analysis.EventDataExtractor import EventDataExtractor

def find_significant_moves(symbol: str, start_date: datetime, end_date: datetime, debug: bool = True):
    """Find significant price moves in the data."""
    print(f"\nFinding significant moves for {symbol}")
    print(f"Period: {start_date} to {end_date}")
    
    # Initialize components
    hdm = HistoricalDataManager(debug=debug)
    validator = DataValidator(debug=debug)
    preprocessor = DataPreprocessor(cache_dir="cache", debug=debug)
    analyzer = StatisticalAnalyzer(debug=debug)
    
    # Load and process data
    print("\nLoading data...")
    price_data = hdm.get_price_data(symbol, start_date, end_date)
    raw_news_data = hdm.get_news_data(symbol, start_date, end_date)
    
    print("Validating news data...")
    _, _, valid_news_data = validator.validate_news_data(raw_news_data)
    
    print("Processing timeframes...")
    preprocessor.align_all_timeframes(price_data, valid_news_data, symbol)
    
    # Find moves
    print("\nAnalyzing price moves...")
    daily_moves = analyzer.get_largest_moves(preprocessor, '1d', n_moves=10)
    hourly_moves = analyzer.get_largest_moves(preprocessor, '1h', n_moves=10)
    
    # Combine and sort all moves
    all_moves = daily_moves + hourly_moves
    all_moves.sort(key=lambda x: abs(x.percentage_change), reverse=True)
    
    print(f"Found {len(all_moves)} significant moves")
    return all_moves, preprocessor

def save_event_windows(moves, preprocessor, symbol: str, output_dir: str = "data/aligned/6day", debug: bool = True):
    """Save 6-day windows (3 days before and after) around each significant move."""
    print(f"\nSaving event windows to {output_dir}")
    
    # Initialize extractor with 3-day windows
    extractor = EventDataExtractor(
        aligned_dir=output_dir,
        lookback_days=3,    # 3 days before
        lookforward_days=3, # 3 days after
        debug=debug
    )
    saved_paths = []
    
    for i, move in enumerate(moves, 1):
        print(f"\nProcessing move {i}/{len(moves)}")
        print(f"Date: {move.date}")
        print(f"Change: {move.percentage_change:.2f}%")
        
        try:
            save_path = extractor.extract_and_save_event(
                event_date=move.date,
                symbol=symbol,
                preprocessor=preprocessor,
                event_number=i
            )
            
            if save_path and save_path.exists():
                print(f"✓ Saved event {i}")
                saved_paths.append(save_path)
            else:
                print(f"✗ Failed to save event {i}")
                
        except Exception as e:
            print(f"Error saving event {i}: {e}")
            continue
    
    print(f"\nSuccessfully saved {len(saved_paths)} of {len(moves)} events")
    return saved_paths

def main():
    """Main execution function."""
    # Parameters
    symbol = 'NVDA'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    output_dir = "data/aligned/6day"  # Changed to 6day
    debug = True
    
    try:
        # First find the moves
        moves, preprocessor = find_significant_moves(symbol, start_date, end_date, debug)
        
        # Then save the event windows
        saved_paths = save_event_windows(moves, preprocessor, symbol, output_dir, debug)
        
        print("\nProcessing complete!")
        print(f"Data saved in: {output_dir}/{symbol}/5min/")
        
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()