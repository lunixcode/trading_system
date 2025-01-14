import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from newsAnalysisNew import NewsAnalysis
from backtester import Backtester, NewsStrategy

class NewsController:
    def __init__(self, 
                 symbol: str,
                 model_choice: str = "claude",
                 window_type: str = "2week"):
        self.symbol = symbol
        self.model_choice = model_choice
        self.window_type = window_type
        self.news_analyzer = NewsAnalysis(symbol=symbol, model_choice=model_choice)
        self.backtester = Backtester(data_dir="data", window_type=window_type)
        
    async def analyze_event_window(self, 
                                 event_data: pd.DataFrame,
                                 metadata: Dict) -> Dict[datetime, Dict]:
        """Analyze all news in an event window"""
        trade_signals = {}
        
        # Process each bar with news
        for idx, row in event_data.iterrows():
            if row['news_count'] > 0 and pd.notna(row['news_indices']):
                indices = [int(i) for i in str(row['news_indices']).split(',')]
                
                for news_idx in indices:
                    if news_idx < len(metadata['news_data']):
                        # Get news item
                        news_item = metadata['news_data'][news_idx]
                        
                        # Analyze news
                        analysis = await self.news_analyzer.analyze_news(news_item)
                        
                        # Store analysis result with timestamp
                        if analysis.get('impact_score', 0) >= 6:
                            trade_signals[pd.to_datetime(idx)] = analysis
                            
        return trade_signals
        
    async def run_event_backtest(self, event_id: int) -> Optional[Dict]:
        """Run complete analysis and backtest for a single event"""
        # Load event data
        data, metadata = self.backtester.load_event_window(event_id, self.symbol)
        if data is None:
            return None
            
        print(f"\nAnalyzing event {event_id}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Analyze news and generate signals
        trade_signals = await self.analyze_event_window(data, metadata)
        
        if trade_signals:
            print(f"Generated {len(trade_signals)} trade signals")
            
            # Run backtest with signals
            results = self.backtester.run_event_backtest(
                event_data=data,
                strategy_class=NewsStrategy,
                strategy_params={
                    'trade_signals': trade_signals,
                    'symbol': self.symbol,
                    'metadata': metadata
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

async def main():
    # Example usage
    controller = NewsController(
        symbol="NVDA",
        model_choice="claude",
        window_type="2week"  # or "6day"
    )
    
    results = await controller.run_sequential_backtest(
        start_event=1,
        end_event=10  # Optional: limit number of events
    )
    
    print("\nBacktest Results:")
    print(f"Total Events: {results['total_events']}")
    print(f"Total Trades: {results['total_trades']}")
   
   #NOT FINISHED, CLAUDES TIMED OUT URGH