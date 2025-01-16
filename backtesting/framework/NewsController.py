import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from backtester import Backtester
from NewsStrategy import NewsStrategy  # Changed to match your file naming
from DataPreprocessor import DataPreprocessor
from DataValidator import DataValidator
from HistoricalDataManager import HistoricalDataManager
from newsAnalysisNew import NewsAnalysis


class NewsController:
    def __init__(self, 
                 symbol: str,
                 model_choice: str = "claude",
                 window_type: str = "2week"):
        self.symbol = symbol
        self.model_choice = model_choice
        self.window_type = window_type
        # Initialize all components once
        self.news_analyzer = NewsAnalysis(symbol=symbol, model_choice=model_choice)
        self.backtester = Backtester(data_dir="data", window_type=window_type)
        self.hdm = HistoricalDataManager(debug=True)
        self.validator = DataValidator(debug=True)
        self.preprocessor = DataPreprocessor(cache_dir="cache", debug=True)
        
    async def analyze_event_window(self, 
                          event_data: pd.DataFrame,
                          metadata: Dict,
                          news_data: List[dict]) -> Dict[datetime, Dict]:
        """Analyze all news in an event window"""
        print(f"\n{'='*50}")
        print(f"ANALYZING EVENT WINDOW")
        print(f"{'='*50}")
        print(f"Event date: {metadata['event_date']}")
        print(f"Number of bars: {len(event_data)}")
        print(f"Number of news items: {len(news_data)}")
        
        trade_signals = {}
        
        # Process each bar with news
        for idx, row in event_data.iterrows():
            if row['news_count'] > 0 and pd.notna(row['news_indices']):
                indices = [int(i) for i in str(row['news_indices']).split(',')]
                print(f"\n{'*'*50}")
                print(f"Found {len(indices)} news items at {idx}")
                print(f"Current price: {row['Close']}")
                
                for news_idx in indices:
                    if news_idx < len(news_data):
                        news_item = news_data[news_idx]
                        print("\nANALYZING NEWS ITEM")
                        print(f"Title: {news_item.get('title', 'No Title')}")
                        
                        # Analyze with LLM
                        analysis = await self.news_analyzer.analyze_news(news_item)
                        print(f"\nLLM ANALYSIS RESULTS:")
                        print(f"Impact Score: {analysis.get('impact_score', 0)}")
                        
                        if analysis.get('impact_score', 0) >= 6:
                            print("\n🚨 HIGH IMPACT DETECTED - GENERATING TRADE SIGNAL 🚨")
                            print(f"Detailed Analysis Available: {bool(analysis.get('detailed_analysis'))}")
                            if analysis.get('detailed_analysis'):
                                scores = analysis['detailed_analysis'].get('scores', {})
                                print("\nTrade Signal Details:")
                                print(f"Sentiment Score: {scores.get('sentiment', 0)}/10")
                                print(f"Reliability Score: {scores.get('reliability', 0)}/10")
                                print(f"Risk Score: {scores.get('risk', 0)}/10")
                            
                            trade_signals[pd.to_datetime(idx)] = analysis
                        else:
                            print("\nLow impact - no trade signal generated")
        
        print(f"\n{'='*50}")
        print(f"EVENT WINDOW ANALYSIS COMPLETE")
        print(f"Total trade signals generated: {len(trade_signals)}")
        print(f"{'='*50}")
        
        return trade_signals

    async def run_event_backtest(self, event_id: int) -> Optional[Dict]:
        """Run complete analysis and backtest for a single event"""
        # Load event data
        data, metadata = self.backtester.load_event_window(event_id, self.symbol)
        if data is None:
            return None
            
        print(f"\nAnalyzing event {event_id}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Get raw data for the window period once
        start_date = pd.to_datetime(metadata['start_date'])
        end_date = pd.to_datetime(metadata['end_date'])
        
        # Load and process data once
        price_data = self.hdm.get_price_data(self.symbol, start_date, end_date)
        raw_news = self.hdm.get_news_data(self.symbol, start_date, end_date)
        _, _, valid_news = self.validator.validate_news_data(raw_news)
        
        # Align data once
        _, news_data = self.preprocessor.align_all_timeframes(
            price_data, 
            valid_news, 
            self.symbol
        )
        
        # Generate signals using the loaded news data
        trade_signals = await self.analyze_event_window(data, metadata, news_data)
        
        if trade_signals:
            print(f"Generated {len(trade_signals)} trade signals")
            
            # Run backtest with signals and data
            results = self.backtester.run_event_backtest(
                event_data=data,
                strategy_class=NewsStrategy,
                strategy_params={
                    'trade_signals': trade_signals,
                    'symbol': self.symbol,
                    'metadata': metadata,
                    'news_data': news_data  # Pass the news data to strategy
                }
            )
            
            return results
        else:
            print("No trade signals generated")
            return None
    async def run_sequential_backtest(self,
                                    start_event: int = 1,
                                    end_event: Optional[int] = None) -> Dict:
        """Run sequential analysis and backtest through multiple events"""
        all_results = []
        current_event = start_event
        
        while True:
            # Process next event
            results = await self.run_event_backtest(current_event)
            
            if results is None:
                if end_event and current_event > end_event:
                    break
            else:
                results['event_id'] = current_event
                all_results.append(results)
                
            current_event += 1
            
        return self.backtester._aggregate_results(all_results)

    def _print_event_summary(self, event_id: int, results: Dict):
        """Print summary of event backtest results"""
        print(f"\nEvent {event_id} Summary:")
        print(f"Trades Executed: {results['strategy_metrics']['total_trades']}")
        print(f"Event PnL: ${results['strategy_metrics']['total_pnl']:.2f}")
        print(f"Return: {results['return_pct']:.2%}")
        
        if results['strategy_metrics']['total_trades'] > 0:
            win_rate = (results['strategy_metrics']['winning_trades'] / 
                       results['strategy_metrics']['total_trades'])
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Max Drawdown: {results['strategy_metrics']['max_drawdown']:.2%}")
    
    def _print_final_summary(self, results: Dict):
        """Print summary of all backtest results"""
        if not results:
            print("\nNo results generated from backtest")
            return
            
        print("\n" + "="*50)
        print("Final Backtest Results")
        print("="*50)
        
        total_events = len(results.get('event_results', []))
        print(f"Total Events Analyzed: {total_events}")
        
        if total_events > 0:
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Total PnL: ${results.get('total_pnl', 0):.2f}")
            print(f"Average Return per Event: {results.get('avg_return_per_event', 0):.2%}")
            
            profitable_events = sum(1 for r in results.get('event_results', []) 
                                if r.get('strategy_metrics', {}).get('total_pnl', 0) > 0)
            print(f"Profitable Events: {profitable_events}/{total_events} "
                f"({profitable_events/total_events:.2%})")
        
            print("\nEvent Breakdown:")
            for event in results.get('event_results', []):
                print(f"\nEvent {event.get('event_id', 'Unknown')}:")
                print(f"Date: {event.get('event_date', 'Unknown')}")
                print(f"PnL: ${event.get('strategy_metrics', {}).get('total_pnl', 0):.2f}")
                print(f"Trades: {event.get('strategy_metrics', {}).get('total_trades', 0)}")

async def main():
    """Example usage"""
    try:
        print("\n" + "="*50)
        print("News-Based Trading Backtest")
        print("="*50)

        # Initialize controller
        controller = NewsController(
            symbol="NVDA",
            model_choice="claude",
            window_type="2week"  # or "6day"
        )
        
        # Run backtest
        results = await controller.run_sequential_backtest(
            start_event=1,
            end_event=10  # Optional: limit number of events
        )
        
        # Print final results
        controller._print_final_summary(results)
        
    except Exception as e:
        print(f"Error in backtest execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())