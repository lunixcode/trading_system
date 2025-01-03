# test_market_sentiment.py
import asyncio
import json
from datetime import datetime
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator
from DataPreprocessor import DataPreprocessor
from marketsentimentanalyzer import MarketSentimentAnalyzer

def print_timeframe_results(timeframe: str, data: dict):
    """Pretty print results for a timeframe"""
    print(f"\n{timeframe} Analysis:")
    print("-" * 30)
    print(f"Total Significant Moves: {data['total_significant_moves']}")
    
    if 'stats' in data and data['stats']:
        stats = data['stats']
        print(f"Average Move Size: {stats.get('avg_move_size', 0):.2f}%")
        print(f"Average News Volume: {stats.get('avg_news_volume', 0):.1f}")
        
        if stats.get('sentiment_move_correlation') is not None:
            print(f"Sentiment-Move Correlation: {stats['sentiment_move_correlation']:.3f}")
        else:
            print("Sentiment-Move Correlation: No correlation data")
            
        print(f"Average Pre-Move Sentiment: {stats.get('avg_pre_sentiment', 0):.2f}")
        print(f"Average Post-Move Sentiment: {stats.get('avg_post_sentiment', 0):.2f}")
        print(f"Average Sentiment Shift: {stats.get('avg_sentiment_shift', 0):.2f}")

async def run_sentiment_analysis(debug_mode: bool = False):
    """Run the sentiment analysis with optional debug mode"""
    # Initialize components
    hdm = HistoricalDataManager()
    validator = DataValidator()
    preprocessor = DataPreprocessor()
    
    # Set test parameters
    symbol = input("\nEnter stock symbol to analyze [default: NVDA]: ").strip() or "NVDA"
    
    # Get date range
    while True:
        try:
            start_date_str = input("Enter start date (YYYY-MM-DD) [default: 2024-01-01]: ").strip() or "2024-01-01"
            end_date_str = input("Enter end date (YYYY-MM-DD) [default: 2024-01-31]: ").strip() or "2024-01-31"
            
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            
            if start_date < end_date:
                break
            else:
                print("Start date must be before end date. Please try again.")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD format.")
    
    print(f"\nAnalyzing {symbol} from {start_date.date()} to {end_date.date()}")
    print("=" * 50)
    
    try:
        # Get data
        print("\nFetching price and news data...")
        price_data = hdm.get_price_data(symbol, start_date, end_date)
        news_data = hdm.get_news_data(symbol, start_date, end_date)
        
        print("\nValidating news data...")
        _, _, valid_news = validator.validate_news_data(news_data)
        
        print("\nPreprocessing data...")
        aligned_data, processed_news = preprocessor.align_all_timeframes(price_data, valid_news)
        
        # Initialize analyzer with debug mode
        analyzer = MarketSentimentAnalyzer(symbol, debug=debug_mode)
        
        # Run analysis
        print("\nRunning sentiment analysis...")
        results = await analyzer.analyze_moves_and_sentiment(
            price_data=aligned_data['5min'],
            analyzed_news=processed_news
        )
        
        # Print results by timeframe
        for timeframe, data in results['price_moves'].items():
            print_timeframe_results(timeframe, data)
            
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"sentiment_analysis_{symbol}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, default=str, indent=4)
            
        print(f"\nDetailed results saved to {output_file}")
        
        return results
        
    except Exception as e:
        print(f"\nError in sentiment analysis: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("\nMarket Sentiment Analysis Test")
        print("=" * 50)
        
        # Ask for debug mode
        debug_input = input("\nEnable debug mode? (y/n) [default: n]: ").strip().lower()
        debug_mode = debug_input == 'y'
        
        # Run the async analysis
        results = asyncio.run(run_sentiment_analysis(debug_mode))
        
    except Exception as e:
        print(f"Error in test execution: {str(e)}")
        raise