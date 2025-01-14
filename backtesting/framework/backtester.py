import pandas as pd
import backtrader as bt
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class PandasNewsData(bt.feeds.PandasData):
    """Custom data feed for price data with news information"""
    lines = ('news_count',)
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('news_count', 'news_count'),
    )

class Backtester:
    def __init__(self, 
                 data_dir: str,
                 window_type: str = "2week",  # "6day" or "2week"
                 initial_cash: float = 100000.0):
        self.data_dir = Path(data_dir)
        self.window_type = window_type
        self.initial_cash = initial_cash
        self.cerebro = bt.Cerebro()
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Backtrader"""
        df = df.copy()
        
        # Handle datetime index
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Convert numeric columns
        numeric_columns = {
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'Volume': 'float64',
            'news_count': 'int32'
        }
        
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        
        return df.fillna(method='ffill').fillna(method='bfill')
        
    def load_event_window(self, event_id: int, symbol: str) -> tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Load a single event window data and metadata"""
        try:
            data_path = self.data_dir / "aligned" / self.window_type / symbol / "5min"
            parquet_file = data_path / f"{event_id}.parquet"
            meta_file = data_path / f"{event_id}_meta.json"
            
            if not parquet_file.exists() or not meta_file.exists():
                return None, None
                
            # Load parquet data
            df = pd.read_parquet(parquet_file)
            
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                
            return df, metadata
            
        except Exception as e:
            print(f"Error loading event {event_id}: {str(e)}")
            return None, None
            
    def run_event_backtest(self, 
                          event_data: pd.DataFrame,
                          strategy_class: bt.Strategy,
                          strategy_params: Dict) -> Dict:
        """Run backtest for a single event window"""
        # Reset Cerebro instance
        self.cerebro = bt.Cerebro()
        
        # Prepare data
        prepared_data = self.prepare_data(event_data)
        data_feed = PandasNewsData(
            dataname=prepared_data,
            fromdate=prepared_data.index.min(),
            todate=prepared_data.index.max()
        )
        
        # Add data and strategy
        self.cerebro.adddata(data_feed)
        self.cerebro.addstrategy(strategy_class, **strategy_params)
        
        # Set broker parameters
        self.cerebro.broker.setcash(self.initial_cash)
        
        # Run backtest
        try:
            initial_value = self.cerebro.broker.getvalue()
            results = self.cerebro.run()
            final_value = self.cerebro.broker.getvalue()
            
            return {
                'initial_value': initial_value,
                'final_value': final_value,
                'return_pct': ((final_value - initial_value) / initial_value) * 100,
                'strategy_metrics': results[0].get_metrics() if results else None
            }
            
        except Exception as e:
            print(f"Backtest error: {str(e)}")
            return None
            
    def run_sequential_events(self,
                            symbol: str,
                            strategy_class: bt.Strategy,
                            start_event: int = 1,
                            end_event: Optional[int] = None,
                            **strategy_params) -> Dict:
        """Run sequential backtest through multiple event windows"""
        all_results = []
        current_event = start_event
        
        while True:
            # Load next event window
            data, metadata = self.load_event_window(current_event, symbol)
            if data is None or (end_event and current_event > end_event):
                break
                
            print(f"\nProcessing event {current_event}")
            print(f"Event date: {metadata.get('event_date')}")
            
            # Run backtest for this event
            event_results = self.run_event_backtest(
                event_data=data,
                strategy_class=strategy_class,
                strategy_params={**strategy_params, 'metadata': metadata}
            )
            
            if event_results:
                event_results['event_id'] = current_event
                event_results['event_date'] = metadata.get('event_date')
                all_results.append(event_results)
                
            current_event += 1
            
        return self._aggregate_results(all_results)
        
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple event windows"""
        if not results:
            return {}
            
        total_trades = sum(r.get('strategy_metrics', {}).get('total_trades', 0) for r in results)
        total_pnl = sum(r.get('strategy_metrics', {}).get('total_pnl', 0) for r in results)
        
        return {
            'total_events': len(results),
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'avg_return_per_event': sum(r['return_pct'] for r in results) / len(results),
            'event_results': results
        }