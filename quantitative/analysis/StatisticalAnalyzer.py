import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv
import requests
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the classes from the framework folder
from backtesting.framework.HistoricalDataManager import HistoricalDataManager
from backtesting.framework.DataValidator import DataValidator

# Import DataPreprocessor type hint only for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backtesting.framework.DataPreprocessor import DataPreprocessor

@dataclass
class PriceMove:
    date: datetime
    timeframe: str
    percentage_change: float
    absolute_change: float
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: float
    news_count: int
    news_indices: str

class StatisticalAnalyzer:
    def __init__(self, cache_dir: str = "cache", debug: bool = False):
        self.debug = debug
        self.cache_dir = Path(cache_dir)
        
    def calculate_price_moves(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various price movement metrics for the dataset."""
        moves_df = df.copy()
        
        # Calculate absolute and percentage changes
        moves_df['absolute_change'] = moves_df['Close'] - moves_df['Open']
        moves_df['percentage_change'] = ((moves_df['Close'] - moves_df['Open']) / moves_df['Open']) * 100
        
        # Calculate true range using vectorized operations
        high_low = moves_df['High'] - moves_df['Low']
        high_close_prev = abs(moves_df['High'] - moves_df['Close'].shift(1))
        low_close_prev = abs(moves_df['Low'] - moves_df['Close'].shift(1))
        
        # Combine all components
        moves_df['true_range'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Handle first row where we don't have previous close
        moves_df.loc[moves_df.index[0], 'true_range'] = high_low.iloc[0]
        
        return moves_df

    def get_largest_moves(self, 
                         preprocessor: 'DataPreprocessor',
                         timeframe: str,
                         n_moves: int = 10,
                         by_percentage: bool = True) -> List[PriceMove]:
        """
        Find the n largest price moves in the dataset for a given timeframe.
        """
        if timeframe not in preprocessor.aligned_data:
            raise ValueError(f"No data available for timeframe {timeframe}")
            
        if self.debug:
            print(f"\nAnalyzing largest moves for {timeframe} timeframe")
            
        # Get data for the specified timeframe
        df = preprocessor.aligned_data[timeframe]
        
        # Calculate moves
        moves_df = self.calculate_price_moves(df)
        
        # Sort by specified metric
        sort_column = 'percentage_change' if by_percentage else 'absolute_change'
        # Create absolute value column for sorting
        moves_df['abs_change'] = moves_df[sort_column].abs()
        largest_moves = moves_df.nlargest(n_moves, 'abs_change')
        # Drop the temporary column
        largest_moves = largest_moves.drop('abs_change', axis=1)
        
        # Convert to PriceMove objects
        price_moves = []
        for _, row in largest_moves.iterrows():
            move = PriceMove(
                date=row['Date'],
                timeframe=timeframe,
                percentage_change=row['percentage_change'],
                absolute_change=row['absolute_change'],
                open_price=row['Open'],
                close_price=row['Close'],
                high_price=row['High'],
                low_price=row['Low'],
                volume=row['Volume'],
                news_count=row['news_count'],
                news_indices=row['news_indices'] if pd.notna(row['news_indices']) else ''
            )
            price_moves.append(move)
            
        if self.debug:
            print(f"\nFound {len(price_moves)} significant price moves")
            for move in price_moves:
                print(f"\nDate: {move.date}")
                print(f"{'Percentage' if by_percentage else 'Absolute'} Change: "
                      f"{move.percentage_change:.2f}% (${move.absolute_change:.2f})")
                print(f"News Count: {move.news_count}")
                
        return price_moves

    def get_news_context(self, 
                      preprocessor: 'DataPreprocessor', 
                      move: PriceMove, 
                      lookback_periods: int = 3) -> List[dict]:
        """Get news context for a specific price move."""
        if self.debug:
            print(f"\nAnalyzing news context for move on {move.date}")
            
        # Get news for the specific period
        news_items = []
        if move.news_indices and move.news_indices.strip():
            current_indices = [int(idx) for idx in move.news_indices.split(',')]
            for idx in current_indices:
                if idx < len(preprocessor.news_data):
                    news_item = preprocessor.news_data[idx].copy()
                    news_item['temporal_relationship'] = 'during_move'
                    news_items.append(news_item)
        
        # Get data for the lookback period
        df = preprocessor.aligned_data[move.timeframe]
        # Convert move.date to tz-naive for comparison
        move_date_naive = pd.to_datetime(move.date).tz_localize(None)
        df_dates_naive = df['Date'].dt.tz_localize(None)
        move_idx = df[df_dates_naive == move_date_naive].index[0]
        
        # Look back several periods
        for i in range(1, lookback_periods + 1):
            if move_idx - i >= 0:
                prior_row = df.iloc[move_idx - i]
                if prior_row['news_count'] > 0 and pd.notna(prior_row['news_indices']):
                    prior_indices = [int(idx) for idx in prior_row['news_indices'].split(',')]
                    for idx in prior_indices:
                        if idx < len(preprocessor.news_data):
                            news_item = preprocessor.news_data[idx].copy()
                            news_item['temporal_relationship'] = f'{i}_periods_before'
                            # Convert both datetimes to tz-naive for comparison
                            move_date = pd.to_datetime(move.date).tz_localize(None)
                            news_date = pd.to_datetime(news_item['date']).tz_localize(None)
                            news_item['time_to_move'] = move_date - news_date
                            news_items.append(news_item)
        
        return news_items

import os
import requests
import json
import pandas as pd
from typing import Dict

import os
import requests
import json
import pandas as pd
from typing import Dict

import os
import requests
import json
import pandas as pd
from typing import Dict

class LLMAnalyzer:
    """Class to handle LLM-based news analysis using DeepSeek."""

    def __init__(self):
        # Load API key from environment variables
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

        # Set the correct DeepSeek API URL
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.debug = True  # Enable debug mode for detailed logging

    def analyze(self, analysis: Dict) -> Dict:
        """Analyze news context using DeepSeek LLM."""
        model_name = "deepseek-chat"

        prompt = f"""You are a financial analyst specializing in identifying causal relationships between news events and stock price movements. Analyze the following scenario:

Stock Movement:
- Date: {analysis['date']}
- Price Change: {analysis['percentage_change']:.2f}% ({analysis['move_direction']})
- Absolute Change: ${analysis['absolute_change']:.2f}
- Trading Volume: {analysis['volume']}

Analyze this situation and provide:
1. The most impactful news article(s) and why they were significant
2. Any common themes or narrative threads across the news
3. The likelihood that these news events caused the price movement (high/medium/low)
4. A brief explanation of the cause-effect relationship

Format your response as a JSON object with the following fields:
- key_articles: Array of most impactful articles (include date and title)
- common_themes: Array of identified themes
- causation_likelihood: String (high/medium/low)
- explanation: String
"""

        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a financial analyst expert at analyzing news impact on stock prices."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9,
            "stream": False
        }

        if self.debug:
            print("\n=== API Request Details ===")
            print(f"Endpoint: {self.api_url}")
            print(f"Model: {request_data['model']}")
            print("Prompt: Truncated for readability")

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=request_data
            )

            if self.debug:
                print("\n=== API Response ===")
                print(f"Status Code: {response.status_code}")

            response.raise_for_status()

            # Parse the response
            llm_response = response.json()

            if "choices" in llm_response and len(llm_response["choices"]) > 0:
                content = llm_response["choices"][0]["message"]["content"].strip()
                if self.debug:
                    print(f"Raw Content: {content}")

                # Handle stringified JSON
                if content.startswith("{") and content.endswith("}"):
                    try:
                        parsed_response = json.loads(content)
                        return parsed_response
                    except json.JSONDecodeError as e:
                        print("\nFailed to parse LLM response as JSON:")
                        print(f"Error: {e}")
                        return {
                            "key_articles": [],
                            "common_themes": [],
                            "causation_likelihood": "unknown",
                            "explanation": "Error parsing LLM response",
                            "raw_response": content
                        }
                else:
                    print("\nUnexpected content format:")
                    print(content)
                    return {
                        "key_articles": [],
                        "common_themes": [],
                        "causation_likelihood": "unknown",
                        "explanation": "Unexpected content format",
                        "raw_response": content
                    }
            else:
                print("\nUnexpected response format from API:")
                print(llm_response)
                return {
                    "key_articles": [],
                    "common_themes": [],
                    "causation_likelihood": "unknown",
                    "explanation": "Unexpected response format",
                    "raw_response": llm_response
                }

        except requests.exceptions.HTTPError as e:
            print(f"\nError in LLM analysis: {str(e)}")
            if e.response:
                print(f"Response content: {e.response.content}")
            return {
                "key_articles": [],
                "common_themes": [],
                "causation_likelihood": "unknown",
                "explanation": f"HTTP Error: {e.response.content if e.response else str(e)}"
            }




def main():
    """Example usage of the StatisticalAnalyzer"""
    from datetime import datetime
    from backtesting.framework.DataPreprocessor import DataPreprocessor
    
    # Initialize components
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    preprocessor = DataPreprocessor(cache_dir="cache", debug=True)
    analyzer = StatisticalAnalyzer(debug=True)
    llm = LLMAnalyzer()
    
    # Load and process data
    symbol = 'NVDA'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Load initial data
    price_data = hdm.get_price_data(symbol, start_date, end_date)
    raw_news_data = hdm.get_news_data(symbol, start_date, end_date)
    _, _, valid_news_data = validator.validate_news_data(raw_news_data)
    
    # Process all timeframes with caching
    preprocessor.align_all_timeframes(price_data, valid_news_data, symbol)
    
    # Analyze largest moves
    print("\nAnalyzing Daily Moves:")
    daily_moves = analyzer.get_largest_moves(preprocessor, '1d', n_moves=10)
    
    # Get initial analysis
    daily_analysis = []
    for move in daily_moves:
        analysis = {
            'date': move.date,
            'timeframe': move.timeframe,
            'percentage_change': move.percentage_change,
            'absolute_change': move.absolute_change,
            'move_direction': 'up' if move.percentage_change > 0 else 'down',
            'volume': move.volume,
            'news_during_move': [],
            'news_before_move': [],
            'has_news_correlation': False,
            'total_news_count': 0
        }
        
        # Get news context
        news_context = analyzer.get_news_context(preprocessor, move, lookback_periods=3)
        if news_context:
            analysis['has_news_correlation'] = True
            analysis['total_news_count'] = len(news_context)
            analysis['news_during_move'] = [n for n in news_context if n['temporal_relationship'] == 'during_move']
            analysis['news_before_move'] = [n for n in news_context if n['temporal_relationship'] != 'during_move']
        
        daily_analysis.append(analysis)
    
    print("\nAnalyzing News Impact:")
    for analysis in daily_analysis:
        if analysis['has_news_correlation']:
            print(f"\nAnalyzing move on {analysis['date']}")
            llm_analysis = llm.analyze(analysis)
            analysis['llm_analysis'] = llm_analysis
            
            print(f"Price Change: {analysis['percentage_change']:.2f}% ({analysis['move_direction']})")
            print(f"Causation Likelihood: {llm_analysis['causation_likelihood']}")
            print("\nKey Articles:")
            for article in llm_analysis['key_articles']:
                print(f"- {article}")
            print("\nCommon Themes:")
            for theme in llm_analysis['common_themes']:
                print(f"- {theme}")
            print(f"\nExplanation: {llm_analysis['explanation']}")
            
    print("\nAnalyzing Hourly Moves:")
    hourly_moves = analyzer.get_largest_moves(preprocessor, '1h', n_moves=10)
    # Similar analysis for hourly moves...

if __name__ == "__main__":
    main()