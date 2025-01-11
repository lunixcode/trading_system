# test_market_sentiment.py
import os
import sys
# Set sys.path to include the 'framework' directory explicitly
framework_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(framework_path)
import asyncio
import json
from datetime import datetime
from HistoricalDataManager import HistoricalDataManager
from DataValidator import DataValidator
from DataPreprocessor import DataPreprocessor
from marketsentimentanalyzer import MarketSentimentAnalyzer
from MetricSaver import MetricSaver

def print_timeframe_results(timeframe: str, data: dict):
    """Pretty print results for a timeframe with proper None handling"""
    print(f"\n{timeframe} Analysis:")
    print("-" * 30)
    
    # Handle total moves
    total_moves = data.get('total_moves', 0)
    significant_moves = len(data.get('significant_moves', []))
    print(f"Total Moves: {total_moves}")
    print(f"Significant Moves: {significant_moves}")
    
    # Handle statistics if they exist
    if 'statistics' in data and data['statistics']:
        stats = data['statistics']
        print("\nStatistics:")
        
        # Handle average move size
        move_size = stats.get('avg_move_size')
        print(f"Average Move Size: {move_size:.2f}%" if move_size is not None else "Average Move Size: No data")
        
        # Handle news volume
        news_vol = stats.get('avg_news_volume')
        print(f"Average News Volume: {news_vol:.1f}" if news_vol is not None else "Average News Volume: No data")
        
        # Handle correlation
        correlation = stats.get('sentiment_move_correlation')
        if correlation is not None:
            print(f"Sentiment-Move Correlation: {correlation:.3f}")
        else:
            print("Sentiment-Move Correlation: No correlation data")
        
        # Handle pre-move sentiment
        pre_sent = stats.get('avg_pre_sentiment')
        print(f"Average Pre-Move Sentiment: {pre_sent:.2f}" if pre_sent is not None else "Average Pre-Move Sentiment: No data")
        
        # Handle post-move sentiment
        post_sent = stats.get('avg_post_sentiment')
        print(f"Average Post-Move Sentiment: {post_sent:.2f}" if post_sent is not None else "Average Post-Move Sentiment: No data")
        
        # Handle sentiment shift
        sent_shift = stats.get('avg_sentiment_shift')
        print(f"Average Sentiment Shift: {sent_shift:.2f}" if sent_shift is not None else "Average Sentiment Shift: No data")
        
    # Print move details if debug mode
    if 'moves_detail' in data and data['moves_detail']:
        print("\nDetailed Moves:")
        for move in data['moves_detail']:
            print(f"\nMove at {move['timestamp']}:")
            price_data = move.get('price_data', {})
            sentiment_data = move.get('sentiment_data', {})
            
            # Print price data with None handling
            price_change = price_data.get('price_change')
            print(f"Change: {price_change:.2f}%" if price_change is not None else "Change: No data")
            print(f"Type: {price_data.get('move_type', 'Unknown')}")
            
            # Print sentiment data with None handling
            news_vol = sentiment_data.get('news_volume')
            print(f"News Volume: {news_vol}" if news_vol is not None else "News Volume: No data")
            
            pre_sent = sentiment_data.get('pre_move_sentiment')
            print(f"Pre-Move Sentiment: {pre_sent:.2f}" if pre_sent is not None else "Pre-Move Sentiment: No data")
            
            post_sent = sentiment_data.get('post_move_sentiment')
            print(f"Post-Move Sentiment: {post_sent:.2f}" if post_sent is not None else "Post-Move Sentiment: No data")

async def run_sentiment_analysis(debug_mode: bool = False):
    """Run the sentiment analysis with optional debug mode"""
    # Initialize components
    hdm = HistoricalDataManager()
    validator = DataValidator()
    preprocessor = DataPreprocessor()
    metric_saver = MetricSaver()  # Initialize MetricSaver
    
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
        
        # Save results using MetricSaver
        saved_files = metric_saver.save_analysis_results(
            results=results,
            symbol=symbol,
            analysis_type="complete"
        )
        
        # Add saved files to results
        results['saved_files'] = saved_files
        
        # Print results by timeframe
        for timeframe, data in results['price_moves'].items():
            print_timeframe_results(timeframe, data)
            
        # Print saved file locations
        if saved_files:
            print("\nResults saved to:")
            for file_type, filepath in saved_files.items():
                print(f"{file_type}: {filepath}")
        
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