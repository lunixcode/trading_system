import os
import sys
from typing import Tuple
from pathlib import Path

class FrameworkComponents:
    @staticmethod
    def initialize_components(
        cache_dir: str = "cache",
        debug: bool = True
    ) -> Tuple:
        """Initialize all framework components in the correct order."""
        # Import locally to avoid circular imports
        from .HistoricalDataManager import HistoricalDataManager
        from .DataValidator import DataValidator
        from .DataPreprocessor import DataPreprocessor
        
        # Create cache directory if it doesn't exist
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize in dependency order
        hdm = HistoricalDataManager(debug=debug)
        validator = DataValidator(debug=debug)
        preprocessor = DataPreprocessor(cache_dir=cache_dir, debug=debug)
        
        return hdm, validator, preprocessor