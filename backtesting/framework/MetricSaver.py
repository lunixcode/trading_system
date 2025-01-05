# Standard library imports
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class MetricSaver:
    """Handles saving of market metrics and analysis results to organized JSON files"""
    
    def __init__(self, base_path: str = "data"):
        """
        Initialize the MetricSaver with a base path for data storage
        
        Args:
            base_path (str): Base directory for storing metric files
        """
        self.base_path = Path(base_path)
        self._ensure_directory_structure()
        
    def _ensure_directory_structure(self) -> None:
        """Create necessary directory structure if it doesn't exist"""
        # Create main directories
        directories = [
            self.base_path,
            self.base_path / "price_moves",
            self.base_path / "sentiment",
            self.base_path / "correlations"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _generate_filename(self, symbol: str, metric_type: str, timeframe: str) -> str:
        """Generate standardized filename for metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{symbol}_{metric_type}_{timeframe}_{timestamp}.json"
        
    def save_analysis_results(self, 
                            results: Dict[str, Any], 
                            symbol: str,
                            analysis_type: str = "complete") -> Dict[str, str]:
        """
        Save complete analysis results with proper organization
        
        Args:
            results: Analysis results dictionary
            symbol: Stock symbol
            analysis_type: Type of analysis ("complete", "price", "sentiment", "correlation")
            
        Returns:
            Dict with saved file locations
        """
        saved_files = {}
        
        try:
            # Save complete results
            if analysis_type == "complete":
                filename = self._generate_filename(symbol, "complete_analysis", "all")
                filepath = self.base_path / filename
                
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=4)
                saved_files['complete'] = str(filepath)
                
            # Save individual timeframe results
            for timeframe, timeframe_data in results.get('price_moves', {}).items():
                # Save price moves
                moves_filename = self._generate_filename(symbol, "price_moves", timeframe)
                moves_filepath = self.base_path / "price_moves" / moves_filename
                
                with open(moves_filepath, 'w') as f:
                    json.dump(timeframe_data, f, indent=4)
                saved_files[f'price_moves_{timeframe}'] = str(moves_filepath)
                
                # Save sentiment analysis if available
                if 'sentiment_analysis' in results and timeframe in results['sentiment_analysis']:
                    sentiment_filename = self._generate_filename(symbol, "sentiment", timeframe)
                    sentiment_filepath = self.base_path / "sentiment" / sentiment_filename
                    
                    with open(sentiment_filepath, 'w') as f:
                        json.dump(results['sentiment_analysis'][timeframe], f, indent=4)
                    saved_files[f'sentiment_{timeframe}'] = str(sentiment_filepath)
            
            return saved_files
            
        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")
            return {}
            
    def save_significant_moves(self, 
                             moves: Dict[str, Any], 
                             symbol: str, 
                             timeframe: str) -> Optional[str]:
        """Save significant price moves data"""
        try:
            filename = self._generate_filename(symbol, "significant_moves", timeframe)
            filepath = self.base_path / "price_moves" / filename
            
            with open(filepath, 'w') as f:
                json.dump(moves, f, indent=4)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error saving significant moves: {str(e)}")
            return None
            
    def save_sentiment_metrics(self, 
                             sentiment_data: Dict[str, Any], 
                             symbol: str, 
                             timeframe: str) -> Optional[str]:
        """Save sentiment metrics data"""
        try:
            filename = self._generate_filename(symbol, "sentiment_metrics", timeframe)
            filepath = self.base_path / "sentiment" / filename
            
            with open(filepath, 'w') as f:
                json.dump(sentiment_data, f, indent=4)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error saving sentiment metrics: {str(e)}")
            return None
            
    def load_analysis_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load saved analysis results"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading analysis results: {str(e)}")
            return None
            
    def get_latest_analysis(self, 
                          symbol: str, 
                          metric_type: str = "complete_analysis", 
                          timeframe: str = "all") -> Optional[Dict[str, Any]]:
        """Get the most recent analysis results for a symbol"""
        try:
            # Find relevant directory
            if metric_type == "complete_analysis":
                search_dir = self.base_path
            else:
                search_dir = self.base_path / metric_type
            
            # Find matching files
            pattern = f"{symbol}_{metric_type}_{timeframe}_*.json"
            matching_files = list(search_dir.glob(pattern))
            
            if not matching_files:
                return None
                
            # Get most recent file
            latest_file = max(matching_files, key=os.path.getctime)
            return self.load_analysis_results(str(latest_file))
            
        except Exception as e:
            print(f"Error getting latest analysis: {str(e)}")
            return None