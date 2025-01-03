# Standard library imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class SentimentMetrics:
    """Store sentiment analysis metrics"""
    period_start: datetime
    period_end: datetime
    avg_sentiment: float
    news_volume: int
    significant_news: int  # news with high impact scores
    sentiment_momentum: float  # rate of sentiment change
    dominant_category: str
    volatility: float

class MarketSentimentAnalyzer:
    """Analyzes market sentiment patterns and their correlation with price movements"""
    
    def __init__(self, symbol: str, debug: bool = False):
        self.symbol = symbol
        self.sentiment_history = pd.DataFrame()
        self.price_history = pd.DataFrame()
        self.significant_moves = pd.DataFrame()
        self.pattern_cache = {}
        self.timeframes = ['5min', '15min', '30min', '1h', '4h', '1D']
        self.debug = debug

    def _debug_print(self, message: str, data: Optional[Any] = None):
        """Debug print helper"""
        if self.debug:
            print(f"\nDEBUG: {message}")
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    print(f"Shape: {data.shape}")
                    print("\nFirst few rows:")
                    print(data.head())
                elif isinstance(data, dict):
                    print("\nDictionary contents:")
                    for key, value in data.items():
                        print(f"{key}: {value}")
                else:
                    print(f"Data: {data}")

    def _prepare_price_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure price data has correct datetime index with UTC timezone"""
        df = price_data.copy()
        self._debug_print("Preparing price data", df)
        
        # Check if we need to convert the index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Look for date/time column
            date_cols = [col for col in df.columns if any(
                time_word in col.lower() 
                for time_word in ['date', 'time', 'datetime']
            )]
            
            if date_cols:
                self._debug_print(f"Found date column: {date_cols[0]}")
                # Use the first found date column
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], utc=True)
                df.set_index(date_cols[0], inplace=True)
            else:
                raise ValueError("No datetime column found in price data")
        else:
            # Convert existing DatetimeIndex to UTC if it's not already
            self._debug_print("Converting existing index to UTC")
            df.index = df.index.tz_localize('UTC' if df.index.tz is None else None)
        
        self._debug_print("Price data prepared", df)
        return df.sort_index()

    def _calculate_price_moves(self, price_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate percentage price moves for given timeframe"""
        self._debug_print(f"\nCalculating price moves for {timeframe}")
        
        # Ensure price data has proper datetime index with UTC timezone
        price_data = self._prepare_price_data(price_data)
        
        # Resample to desired timeframe
        resampled = price_data.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        self._debug_print("Resampled data", resampled)
        
        # Calculate percentage moves
        resampled['price_change'] = resampled['Close'].pct_change() * 100
        resampled['high_low_range'] = ((resampled['High'] - resampled['Low']) / resampled['Low']) * 100
        resampled['move_type'] = resampled['price_change'].apply(lambda x: 'up' if x > 0 else 'down')
        
        self._debug_print("Calculated moves", resampled)
        return resampled
    
    def _find_significant_moves(self, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Find top 10 price moves for each timeframe"""
        self._debug_print("\nFinding significant moves")
        significant_moves = {}
        
        for tf in self.timeframes:
            self._debug_print(f"\nProcessing timeframe: {tf}")
            # Calculate moves for timeframe
            moves_df = self._calculate_price_moves(price_data, tf)
            
            # Find top 10 up and down moves
            top_up = moves_df[moves_df['price_change'] > 0].nlargest(10, 'price_change')
            top_down = moves_df[moves_df['price_change'] < 0].nsmallest(10, 'price_change')
            
            self._debug_print(f"Found {len(top_up)} up moves and {len(top_down)} down moves")
            
            # Combine and sort by absolute value of price change
            top_moves = pd.concat([top_up, top_down])
            top_moves['abs_change'] = abs(top_moves['price_change'])
            top_moves = top_moves.sort_values('abs_change', ascending=False)
            
            # Add additional metrics
            top_moves['volume_change'] = top_moves['Volume'].pct_change()
            top_moves['volatility'] = (top_moves['High'] - top_moves['Low']) / top_moves['Open']
            
            significant_moves[tf] = top_moves
            self._debug_print(f"Significant moves for {tf}", top_moves)

        return significant_moves

    def _calculate_sentiment_metrics(self, analyzed_news: List[Dict]) -> pd.DataFrame:
        """Calculate sentiment metrics for different timeframes"""
        self._debug_print(f"Processing {len(analyzed_news)} news items")
        
        if not analyzed_news:
            return pd.DataFrame()
            
        # Debug the raw news data first
        self._debug_print("Sample of raw news items:", analyzed_news[:2])
            
        news_df = pd.DataFrame(analyzed_news)
        self._debug_print("Initial news dataframe columns:", news_df.columns.tolist())
        
        # Time field handling
        time_field = next((field for field in ['date', 'timestamp', 'time'] 
                          if field in news_df.columns), None)
        
        if not time_field:
            self._debug_print("No timestamp field found")
            return pd.DataFrame()
        
        # Debug available scores before processing
        if 'sentiment_score' in news_df.columns:
            self._debug_print("Original sentiment scores:", news_df['sentiment_score'].value_counts().head())
        if 'impact_score' in news_df.columns:
            self._debug_print("Original impact scores:", news_df['impact_score'].value_counts().head())
            
        news_df[time_field] = pd.to_datetime(news_df[time_field], utc=True)
        news_df.set_index(time_field, inplace=True)
        
        # Column handling with debug info
        required_columns = {
            'sentiment_score': 'impact_score',
            'category': 'category',
            'impact_score': 'impact_score'
        }
        
        for col, default_col in required_columns.items():
            if col not in news_df.columns and default_col in news_df.columns:
                self._debug_print(f"Using {default_col} for {col}")
                news_df[col] = news_df[default_col]
                self._debug_print(f"Values after setting {col}:", 
                                news_df[col].value_counts().head())
            elif col not in news_df.columns:
                self._debug_print(f"No data for {col}, using default 0")
                news_df[col] = 0
        
        # Debug final dataframe state before processing metrics
        self._debug_print("Final dataframe columns:", news_df.columns.tolist())
        self._debug_print("Final sentiment score stats:", 
                         news_df['sentiment_score'].describe())
                
        return self._process_metrics(news_df)

    def _process_metrics(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Process metrics after initial setup"""
        metrics_list = []
        
        for tf in self.timeframes:
            self._debug_print(f"\nProcessing timeframe: {tf}")
            grouped = news_df.resample(tf)
            
            for name, group in grouped:
                if not group.empty:
                    self._debug_print(f"\nGroup at {name} data:")
                    self._debug_print("Group sentiment scores:", group['sentiment_score'].values)
                    
                    avg_sent = group['sentiment_score'].mean()
                    self._debug_print(f"Group at {name}, avg sentiment: {avg_sent}")
                    self._debug_print(f"Group size: {len(group)}")
                    
                    metrics = SentimentMetrics(
                        period_start=group.index[0],
                        period_end=group.index[-1],
                        avg_sentiment=avg_sent,
                        news_volume=len(group),
                        significant_news=len(group[group['impact_score'] >= 7]),
                        sentiment_momentum=group['sentiment_score'].diff().mean(),
                        dominant_category=group['category'].mode()[0] if 'category' in group else 'unknown',
                        volatility=group['sentiment_score'].std()
                    )
                    metrics_list.append(vars(metrics))
        
        metrics_df = pd.DataFrame(metrics_list)
        if not metrics_df.empty:
            metrics_df['period_start'] = pd.to_datetime(metrics_df['period_start'], utc=True)
            metrics_df.set_index('period_start', inplace=True)
            self._debug_print("Final metrics dataframe:", metrics_df)

    def _correlate_with_price_moves(self, 
                                  price_data: pd.DataFrame, 
                                  sentiment_metrics: pd.DataFrame) -> Dict:
        """Find correlations between sentiment and significant price movements"""
        self._debug_print("\nStarting correlation analysis")
        self._debug_print("Sentiment metrics shape:", sentiment_metrics.shape)
        
        # Ensure consistent timezone handling
        price_data = self._prepare_price_data(price_data)
        sentiment_metrics.index = sentiment_metrics.index.tz_localize('UTC' if sentiment_metrics.index.tz is None else None)
        
        # Find significant moves first
        significant_moves = self._find_significant_moves(price_data)
        correlations = {}
        
        for timeframe, moves in significant_moves.items():
            self._debug_print(f"\nAnalyzing {timeframe}")
            correlations[timeframe] = {
                'moves': [],
                'sentiment_context': [],
                'stats': {}
            }
            
            for idx, move in moves.iterrows():
                # Convert idx to UTC if needed
                idx = pd.to_datetime(idx).tz_localize('UTC' if idx.tz is None else None)
                
                # Look for news/sentiment in window around the move
                window_start = idx - pd.Timedelta(hours=24)
                window_end = idx + pd.Timedelta(hours=24)
                
                self._debug_print(f"Analyzing window: {window_start} to {window_end}")
                
                # Find sentiment metrics in this window
                window_sentiment = sentiment_metrics[
                    (sentiment_metrics.index >= window_start) & 
                    (sentiment_metrics.index <= window_end)
                ]
                
                self._debug_print(f"Found {len(window_sentiment)} sentiment records in window")
                
                move_data = {
                    'timestamp': idx,
                    'price_change': move['price_change'],
                    'move_type': move['move_type'],
                    'volume_change': move['volume_change'],
                    'volatility': move['volatility'],
                    'high_low_range': move['high_low_range'],
                    'pre_move_sentiment': None,
                    'post_move_sentiment': None,
                    'sentiment_shift': None,
                    'news_volume': None
                }
                
                if not window_sentiment.empty:
                    # Split sentiment before and after move
                    pre_move = window_sentiment[window_sentiment.index < idx]
                    post_move = window_sentiment[window_sentiment.index >= idx]
                    
                    self._debug_print(f"Pre-move records: {len(pre_move)}")
                    self._debug_print(f"Post-move records: {len(post_move)}")
                    
                    # Calculate sentiment metrics
                    if not pre_move.empty:
                        move_data['pre_move_sentiment'] = pre_move['avg_sentiment'].mean()
                        self._debug_print("Pre-move sentiment:", move_data['pre_move_sentiment'])
                    if not post_move.empty:
                        move_data['post_move_sentiment'] = post_move['avg_sentiment'].mean()
                        self._debug_print("Post-move sentiment:", move_data['post_move_sentiment'])
                    
                    if move_data['pre_move_sentiment'] is not None and move_data['post_move_sentiment'] is not None:
                        move_data['sentiment_shift'] = move_data['post_move_sentiment'] - move_data['pre_move_sentiment']
                        self._debug_print("Sentiment shift:", move_data['sentiment_shift'])
                    
                    move_data['news_volume'] = len(window_sentiment)
                    
                    self._debug_print("Move data:", move_data)
                
                correlations[timeframe]['moves'].append(move_data)
            
            # Calculate statistics for this timeframe
            moves_df = pd.DataFrame(correlations[timeframe]['moves'])
            if not moves_df.empty:
                stats = {}
                try:
                    stats = {
                        'avg_move_size': moves_df['price_change'].abs().mean(),
                        'avg_pre_sentiment': moves_df['pre_move_sentiment'].mean(),
                        'avg_post_sentiment': moves_df['post_move_sentiment'].mean(),
                        'avg_sentiment_shift': moves_df['sentiment_shift'].mean(),
                        'avg_news_volume': moves_df['news_volume'].mean(),
                    }
                    
                    # Calculate correlation only if we have valid pre-move sentiment
                    if 'pre_move_sentiment' in moves_df.columns and moves_df['pre_move_sentiment'].notna().any():
                        valid_data = moves_df.dropna(subset=['price_change', 'pre_move_sentiment'])
                        if not valid_data.empty:
                            stats['sentiment_move_correlation'] = valid_data['price_change'].corr(
                                valid_data['pre_move_sentiment']
                            )
                        else:
                            stats['sentiment_move_correlation'] = None
                    else:
                        stats['sentiment_move_correlation'] = None
                        
                except Exception as e:
                    self._debug_print(f"Error calculating stats for {timeframe}: {str(e)}")
                    stats = {
                        'avg_move_size': 0,
                        'avg_pre_sentiment': 0,
                        'avg_post_sentiment': 0,
                        'avg_sentiment_shift': 0,
                        'avg_news_volume': 0,
                        'sentiment_move_correlation': None
                    }
                
                correlations[timeframe]['stats'] = stats
                self._debug_print(f"Statistics for {timeframe}:", stats)
        
        return correlations

    async def analyze_moves_and_sentiment(self, price_data: pd.DataFrame, analyzed_news: List[Dict]) -> Dict:
        """Main method to analyze price moves and correlate with sentiment"""
        try:
            self._debug_print("\nStarting analysis")
            # Ensure price data has proper datetime index with UTC timezone
            price_data = self._prepare_price_data(price_data)
            
            # Store price data
            self.price_history = price_data
            
            # Calculate sentiment metrics
            sentiment_metrics = self._calculate_sentiment_metrics(analyzed_news)
            
            # Initialize summary structure
            summary = {
                'analysis_period': {
                    'start': price_data.index[0],
                    'end': price_data.index[-1]
                },
                'price_moves': {},
                'news_coverage': {
                    'total_news': len(analyzed_news),
                    'high_impact_news': len([n for n in analyzed_news if n.get('impact_score', 0) >= 7])
                }
            }
            
            self._debug_print("Analysis period:", summary['analysis_period'])
            
            # Only proceed with correlation if we have sentiment metrics
            if not sentiment_metrics.empty:
                correlations = self._correlate_with_price_moves(price_data, sentiment_metrics)
                summary['sentiment_correlations'] = correlations
                
                # Add summary stats for each timeframe
                for tf in self.timeframes:
                    if tf in correlations:
                        summary['price_moves'][tf] = {
                            'total_significant_moves': len(correlations[tf]['moves']),
                            'stats': correlations[tf]['stats'] if 'stats' in correlations[tf] else {}
                        }
            else:
                print("No sentiment metrics available for correlation analysis")
                summary['sentiment_correlations'] = {}
                for tf in self.timeframes:
                    summary['price_moves'][tf] = {
                        'total_significant_moves': 0,
                        'stats': {}
                    }
            
            return summary
            
        except Exception as e:
            print(f"Error in analyze_moves_and_sentiment: {str(e)}")
            raise