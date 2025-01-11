import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the classes from the framework folder
from backtesting.framework.HistoricalDataManager import HistoricalDataManager
from backtesting.framework.DataValidator import DataValidator
from backtesting.framework.DataPreprocessor import DataPreprocessor

def test_classes():
    """
    Test if the imported classes can be instantiated.
    """
    try:
        # Instantiate the classes
        hdm = HistoricalDataManager()
        validator = DataValidator()
        preprocessor = DataPreprocessor()

        # Print success messages
        print("HistoricalDataManager loaded successfully:", hdm)
        print("DataValidator loaded successfully:", validator)
        print("DataPreprocessor loaded successfully:", preprocessor)

    except Exception as e:
        print("Error while testing class imports:", str(e))

# Run the test
if __name__ == "__main__":
    test_classes()
