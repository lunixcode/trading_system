import os
import sys
from datetime import datetime
from typing import Dict, List
import json
import requests
from dotenv import load_dotenv
from pathlib import Path

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import required components
from backtesting.framework.HistoricalDataManager import HistoricalDataManager
from backtesting.framework.DataValidator import DataValidator
from backtesting.framework.DataPreprocessor import DataPreprocessor
from StatisticalAnalyzer import StatisticalAnalyzer

class NewsImpactAnalyzer:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self._setup_environment()
        
    def _setup_environment(self):
        """Initialize API keys and configurations"""
        load_dotenv()
        
        # DeepSeek setup
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

        self.api_url = "https://api.deepseek.com/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def analyze_market_moves(self, moves_data: List[Dict]) -> List[Dict]:
        """Analyze a list of significant market moves with news context."""
        analyzed_moves = []
        
        for move in moves_data:
            if self.debug:
                print(f"\nAnalyzing move on {move['date']}")
                
            if move['has_news_correlation']:
                llm_analysis = self._analyze_move(move)
                move['llm_analysis'] = llm_analysis
                
                if self.debug:
                    self._print_analysis(move, llm_analysis)
                    
            analyzed_moves.append(move)
            
        return analyzed_moves

    def _analyze_move(self, move: Dict) -> Dict:
        """Analyze a single market move using DeepSeek LLM."""
        prompt = self._create_analysis_prompt(move)
        
        try:
            response = self._call_llm_api(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            if self.debug:
                print(f"Error in LLM analysis: {str(e)}")
            return self._create_error_response(str(e))

    def _create_analysis_prompt(self, move: Dict) -> str:
        """Create the analysis prompt for the LLM."""
        return f"""You are a financial analyst specializing in identifying causal relationships between news events and stock price movements. Analyze the following scenario:

Stock Movement:
- Date: {move['date']}
- Price Change: {move['percentage_change']:.2f}% ({move['move_direction']})
- Absolute Change: ${move['absolute_change']:.2f}
- Trading Volume: {move['volume']}

News Context:
During Move: {len(move['news_during_move'])} articles
Before Move: {len(move['news_before_move'])} articles

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

    def _call_llm_api(self, prompt: str) -> Dict:
        """Make the API call to DeepSeek."""
        request_data = {
            "model": "deepseek-chat",
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

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=request_data
        )
        
        response.raise_for_status()
        return response.json()

    def _parse_llm_response(self, response: Dict) -> Dict:
        """Parse the LLM API response."""
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"].strip()
            
            if content.startswith("{") and content.endswith("}"):
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    if self.debug:
                        print(f"Error parsing JSON response: {e}")
                    return self._create_error_response("JSON parse error")
            else:
                return self._create_error_response("Unexpected content format")
        else:
            return self._create_error_response("Invalid API response")

    def _create_error_response(self, error_message: str) -> Dict:
        """Create a standardized error response."""
        return {
            "key_articles": [],
            "common_themes": [],
            "causation_likelihood": "unknown",
            "explanation": f"Error: {error_message}"
        }

    def _print_analysis(self, move: Dict, analysis: Dict):
        """Print the analysis results in a readable format."""
        print(f"\nMove Analysis:")
        print(f"Date: {move['date']}")
        print(f"Price Change: {move['percentage_change']:.2f}% ({move['move_direction']})")
        print(f"Causation Likelihood: {analysis['causation_likelihood']}")
        
        print("\nKey Articles:")
        for article in analysis['key_articles']:
            print(f"- {article}")
            
        print("\nCommon Themes:")
        for theme in analysis['common_themes']:
            print(f"- {theme}")
            
        print(f"\nExplanation: {analysis['explanation']}")

def main():
    """Example usage of the NewsImpactAnalyzer"""
    # Initialize components
    hdm = HistoricalDataManager(debug=True)
    validator = DataValidator(debug=True)
    preprocessor = DataPreprocessor(cache_dir="cache", debug=True)
    stat_analyzer = StatisticalAnalyzer(debug=True)
    news_analyzer = NewsImpactAnalyzer(debug=True)
    
    # Set analysis parameters
    symbol = 'NVDA'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Get and process data
    price_data = hdm.get_price_data(symbol, start_date, end_date)
    raw_news_data = hdm.get_news_data(symbol, start_date, end_date)
    _, _, valid_news_data = validator.validate_news_data(raw_news_data)
    
    # Process timeframes
    preprocessor.align_all_timeframes(price_data, valid_news_data, symbol)
    
    # Get significant moves
    print("\nFinding significant market moves...")
    daily_moves = stat_analyzer.get_largest_moves(preprocessor, '1d', n_moves=10)
    
    # Prepare moves data for analysis
    moves_data = []
    for move in daily_moves:
        move_data = {
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
        news_context = stat_analyzer.get_news_context(preprocessor, move, lookback_periods=3)
        if news_context:
            move_data['has_news_correlation'] = True
            move_data['total_news_count'] = len(news_context)
            move_data['news_during_move'] = [n for n in news_context if n['temporal_relationship'] == 'during_move']
            move_data['news_before_move'] = [n for n in news_context if n['temporal_relationship'] != 'during_move']
        
        moves_data.append(move_data)
    
    # Analyze news impact
    print("\nAnalyzing news impact...")
    analyzed_moves = news_analyzer.analyze_market_moves(moves_data)
    
    # Save results
    output_dir = Path("analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{symbol}_news_impact_{start_date.date()}_{end_date.date()}.json"
    with open(output_file, 'w') as f:
        json.dump(analyzed_moves, f, indent=2, default=str)
        
    print(f"\nAnalysis results saved to {output_file}")

if __name__ == "__main__":
    main()