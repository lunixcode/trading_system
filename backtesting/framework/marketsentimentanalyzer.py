# Standard library imports
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from MetricSaver import MetricSaver

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
        self.metric_saver = MetricSaver()

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
        """Ensure price data has correct datetime index and enhanced metrics"""
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
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Calculate additional price metrics
        try:
            # Basic price metrics
            df['true_range'] = pd.DataFrame({
                'hl': df['High'] - df['Low'],
                'hc': abs(df['High'] - df['Close'].shift(1)),
                'lc': abs(df['Low'] - df['Close'].shift(1))
            }).max(axis=1)
            
            df['price_change'] = df['Close'].pct_change() * 100
            df['price_change_abs'] = df['price_change'].abs()
            df['high_low_range'] = ((df['High'] - df['Low']) / df['Low']) * 100
            df['body_size'] = abs(df['Close'] - df['Open'])
            df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            
            # Volume metrics
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            df['relative_volume'] = df['Volume'] / df['volume_ma']
            
            # Volatility metrics
            df['volatility'] = df['true_range'] / df['Close'].shift(1) * 100
            df['volatility_ma'] = df['volatility'].rolling(window=20).mean()
            
            # Movement classification
            df['candle_type'] = np.where(df['Close'] >= df['Open'], 'bullish', 'bearish')
            df['size_category'] = pd.qcut(df['price_change_abs'], q=4, labels=['small', 'medium', 'large', 'extreme'])
            
            # Add timestamp components for potential patterns
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            
            if self.debug:
                self._debug_print("Added price metrics:", 
                                [col for col in df.columns if col not in price_data.columns])
                self._debug_print("Sample calculations:", df[['price_change', 'volatility', 'true_range']].head())
        
        except Exception as e:
            self._debug_print(f"Error calculating price metrics: {str(e)}")
            # If calculations fail, return cleaned data without additional metrics
            return df.sort_index()
        
        self._debug_print("Price data prepared with enhanced metrics", df)
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
        """Calculate enhanced sentiment metrics for different timeframes"""
        if not analyzed_news:
            print("No news data provided")
            return pd.DataFrame()
            
        # First, flatten the sentiment structure with enhanced metrics
        flattened_news = []
        for news in analyzed_news:
            flat_news = {
                'timestamp': news.get('date'),
                'title': news.get('title', ''),
                'sentiment_metrics': {}
            }
            
            if 'sentiment' in news and isinstance(news['sentiment'], dict):
                sentiment_data = news['sentiment']
                flat_news['sentiment_metrics'] = {
                    'polarity': sentiment_data.get('polarity', 0),
                    'positive_score': sentiment_data.get('pos', 0),
                    'negative_score': sentiment_data.get('neg', 0),
                    'neutral_score': sentiment_data.get('neu', 0),
                    'sentiment_strength': abs(sentiment_data.get('polarity', 0)),  # Absolute strength
                    'sentiment_direction': 1 if sentiment_data.get('polarity', 0) > 0 else -1 if sentiment_data.get('polarity', 0) < 0 else 0
                }
            else:
                flat_news['sentiment_metrics'] = {
                    'polarity': 0,
                    'positive_score': 0,
                    'negative_score': 0,
                    'neutral_score': 0,
                    'sentiment_strength': 0,
                    'sentiment_direction': 0
                }
            
            flattened_news.append(flat_news)
            
        news_df = pd.DataFrame(flattened_news)
        if news_df.empty:
            print("News DataFrame is empty")
            return pd.DataFrame()
        
        # Convert to datetime and set as index
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], utc=True)
        news_df.set_index('timestamp', inplace=True)
        
        # Extract sentiment metrics into separate columns
        for metric in ['polarity', 'positive_score', 'negative_score', 'neutral_score', 'sentiment_strength', 'sentiment_direction']:
            news_df[metric] = news_df['sentiment_metrics'].apply(lambda x: x.get(metric, 0))
            
        # Add rolling metrics
        for window in ['1H', '4H', '1D']:
            news_df[f'sentiment_momentum_{window}'] = news_df['polarity'].rolling(window).mean().diff()
            news_df[f'sentiment_volatility_{window}'] = news_df['polarity'].rolling(window).std()
            news_df[f'news_intensity_{window}'] = news_df['sentiment_strength'].rolling(window).mean()
        
        if self.debug:
            print("\nDEBUG: Sentiment metrics summary:")
            print(news_df.describe())
            print("\nDEBUG: Sample of final processed news data:")
            print(news_df.head())
        
        return self._process_metrics(news_df)

    def _process_metrics(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Process enhanced metrics into structured format"""
        metrics_list = []
        
        for tf in self.timeframes:
            self._debug_print(f"\nProcessing timeframe: {tf}")
            grouped = news_df.resample(tf)
            
            for name, group in grouped:
                if not group.empty:
                    if self.debug:
                        self._debug_print(f"Processing group at {name} with {len(group)} entries")
                    
                    # Basic metrics
                    base_metrics = {
                        'period_start': group.index[0],
                        'period_end': group.index[-1],
                        'timeframe': tf,
                        'group_size': len(group)
                    }
                    
                    # Sentiment averages
                    sentiment_metrics = {
                        'avg_sentiment': group['polarity'].mean(),
                        'weighted_sentiment': (group['polarity'] * group['sentiment_strength']).sum() / group['sentiment_strength'].sum()
                        if group['sentiment_strength'].sum() > 0 else 0,
                        'sentiment_strength': group['sentiment_strength'].mean(),
                        'sentiment_direction': group['sentiment_direction'].mode()[0]
                    }
                    
                    # Sentiment components
                    component_metrics = {
                        'positive_ratio': group['positive_score'].mean(),
                        'negative_ratio': group['negative_score'].mean(),
                        'neutral_ratio': group['neutral_score'].mean()
                    }
                    
                    # Movement metrics
                    movement_metrics = {
                        'sentiment_momentum': group['polarity'].diff().mean(),
                        'sentiment_acceleration': group['polarity'].diff().diff().mean(),
                        'momentum_1H': group['sentiment_momentum_1H'].mean(),
                        'momentum_4H': group['sentiment_momentum_4H'].mean(),
                        'momentum_1D': group['sentiment_momentum_1D'].mean()
                    }
                    
                    # Volatility metrics
                    volatility_metrics = {
                        'sentiment_volatility': group['polarity'].std(),
                        'sentiment_range': group['polarity'].max() - group['polarity'].min(),
                        'volatility_1H': group.get('sentiment_volatility_1H', pd.Series()).mean(),
                        'volatility_4H': group.get('sentiment_volatility_4H', pd.Series()).mean(),
                        'volatility_1D': group.get('sentiment_volatility_1D', pd.Series()).mean()
                    }
                    
                    # News intensity metrics
                    intensity_metrics = {
                        'news_volume': len(group),
                        'news_intensity': group['sentiment_strength'].sum(),
                        'intensity_1H': group.get('news_intensity_1H', pd.Series()).mean(),
                        'intensity_4H': group.get('news_intensity_4H', pd.Series()).mean(),
                        'intensity_1D': group.get('news_intensity_1D', pd.Series()).mean()
                    }
                    
                    # Combine all metrics
                    combined_metrics = {
                        **base_metrics,
                        **sentiment_metrics,
                        **component_metrics,
                        **movement_metrics,
                        **volatility_metrics,
                        **intensity_metrics
                    }
                    
                    # Add news titles for reference
                    combined_metrics['news_titles'] = group['title'].tolist() if 'title' in group else []
                    
                    metrics_list.append(combined_metrics)
                    
                    if self.debug:
                        self._debug_print(f"Processed metrics for {name}", combined_metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        if not metrics_df.empty:
            metrics_df.set_index('period_start', inplace=True)
            
            if self.debug:
                self._debug_print("Final metrics dataframe shape:", metrics_df.shape)
                self._debug_print("Metrics columns:", metrics_df.columns.tolist())
        
        return metrics_df

    def _process_metrics(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Process enhanced metrics into structured format"""
        metrics_list = []

        if self.debug:
            print("\nDEBUG: Available columns in news_df:", news_df.columns.tolist())
            
        for tf in self.timeframes:
            self._debug_print(f"\nProcessing timeframe: {tf}")
            grouped = news_df.resample(tf)
            
            for name, group in grouped:
                if not group.empty:
                    if self.debug:
                        self._debug_print(f"Processing group at {name} with {len(group)} entries")
                    
                    # Basic metrics
                    base_metrics = {
                        'period_start': group.index[0],
                        'period_end': group.index[-1],
                        'timeframe': tf,
                        'group_size': len(group)
                    }
                    
                    # Sentiment averages
                    sentiment_metrics = {
                        'avg_sentiment': group['polarity'].mean() if 'polarity' in group else 0,
                        'weighted_sentiment': (group['polarity'] * group['sentiment_strength']).sum() / group['sentiment_strength'].sum()
                        if 'sentiment_strength' in group and group['sentiment_strength'].sum() > 0 else 0,
                        'sentiment_strength': group['sentiment_strength'].mean() if 'sentiment_strength' in group else 0,
                        'sentiment_direction': group['sentiment_direction'].mode()[0] if 'sentiment_direction' in group else 0
                    }
                    
                    # Sentiment components
                    component_metrics = {
                        'positive_ratio': group['positive_score'].mean() if 'positive_score' in group else 0,
                        'negative_ratio': group['negative_score'].mean() if 'negative_score' in group else 0,
                        'neutral_ratio': group['neutral_score'].mean() if 'neutral_score' in group else 0
                    }
                    
                    # Movement metrics
                    movement_metrics = {
                        'sentiment_momentum': group['polarity'].diff().mean() if 'polarity' in group else 0,
                        'sentiment_acceleration': group['polarity'].diff().diff().mean() if 'polarity' in group else 0
                    }
                    
                    # Add rolling metrics if they exist
                    for window in ['1H', '4H', '1D']:
                        momentum_col = f'sentiment_momentum_{window}'
                        if momentum_col in group:
                            movement_metrics[f'momentum_{window}'] = group[momentum_col].mean()
                    
                    # Volatility metrics
                    volatility_metrics = {
                        'sentiment_volatility': group['polarity'].std() if 'polarity' in group else 0,
                        'sentiment_range': (group['polarity'].max() - group['polarity'].min()) if 'polarity' in group else 0
                    }
                    
                    # News intensity metrics
                    intensity_metrics = {
                        'news_volume': len(group),
                        'news_intensity': group['sentiment_strength'].sum() if 'sentiment_strength' in group else 0
                    }
                    
                    # Combine all metrics
                    combined_metrics = {
                        **base_metrics,
                        **sentiment_metrics,
                        **component_metrics,
                        **movement_metrics,
                        **volatility_metrics,
                        **intensity_metrics
                    }
                    
                    # Add news titles if available
                    if 'title' in group:
                        combined_metrics['news_titles'] = group['title'].tolist()
                    
                    metrics_list.append(combined_metrics)
                    
                    if self.debug:
                        self._debug_print(f"Processed metrics for {name}", combined_metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        if not metrics_df.empty:
            metrics_df.set_index('period_start', inplace=True)
            
            if self.debug:
                self._debug_print("Final metrics dataframe shape:", metrics_df.shape)
                self._debug_print("Metrics columns:", metrics_df.columns.tolist())
        
        return metrics_df

    def _find_significant_moves(self, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Find top 10 price moves for each timeframe with detailed information"""
        self._debug_print("\nFinding significant moves")
        significant_moves = {}
        
        for tf in self.timeframes:
            self._debug_print(f"\nProcessing timeframe: {tf}")
            
            # Calculate moves for timeframe
            moves_df = self._calculate_price_moves(price_data, tf)
            
            if moves_df.empty:
                self._debug_print(f"No price data for timeframe {tf}")
                continue
                
            # Add debug prints for price changes
            self._debug_print("Price changes summary:", moves_df['price_change'].describe())
            
            # Find top 10 up and down moves
            top_up = moves_df[moves_df['price_change'] > 0].nlargest(10, 'price_change')
            top_down = moves_df[moves_df['price_change'] < 0].nsmallest(10, 'price_change')
            
            self._debug_print(f"Found {len(top_up)} up moves and {len(top_down)} down moves")
            
            if len(top_up) == 0 and len(top_down) == 0:
                self._debug_print("No significant moves found")
                continue
                
            # Combine and sort by absolute value of price change
            top_moves = pd.concat([top_up, top_down])
            
            # Enhanced move information
            top_moves['timestamp'] = top_moves.index
            top_moves['timeframe'] = tf
            top_moves['abs_change'] = abs(top_moves['price_change'])
            top_moves['move_type'] = top_moves['price_change'].apply(
                lambda x: 'up' if x > 0 else 'down'
            )
            
            # Add additional metrics
            top_moves['volume_change'] = top_moves['Volume'].pct_change()
            top_moves['volatility'] = (top_moves['High'] - top_moves['Low']) / top_moves['Open']
            top_moves['price_range'] = top_moves['High'] - top_moves['Low']
            top_moves['open_close_range'] = abs(top_moves['Open'] - top_moves['Close'])
            
            # Calculate move duration if possible
            if tf != '1D':  # For intraday timeframes
                top_moves['move_start'] = top_moves.index - pd.Timedelta(tf)
                top_moves['move_end'] = top_moves.index
            
            # Sort by absolute change
            top_moves = top_moves.sort_values('abs_change', ascending=False)
            
            significant_moves[tf] = top_moves
            self._debug_print(f"Significant moves for {tf}", top_moves)

        return significant_moves

    def _correlate_with_price_moves(self, 
                                  price_data: pd.DataFrame, 
                                  sentiment_metrics: pd.DataFrame) -> Dict:
        """Find correlations between sentiment and significant price movements with detailed timing"""
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
                
                # Enhanced move data structure
                move_data = {
                    'timestamp': idx.isoformat(),
                    'timeframe': timeframe,
                    'price_data': {
                        'price_change': float(move['price_change']),
                        'move_type': move['move_type'],
                        'open': float(move['Open']),
                        'high': float(move['High']),
                        'low': float(move['Low']),
                        'close': float(move['Close']),
                        'volume_change': float(move['volume_change']) if pd.notna(move['volume_change']) else None,
                        'volatility': float(move['volatility']),
                        'price_range': float(move['price_range']),
                        'open_close_range': float(move['open_close_range'])
                    },
                    'sentiment_data': {
                        'pre_move_sentiment': None,
                        'post_move_sentiment': None,
                        'sentiment_shift': None,
                        'news_volume': None,
                        'sentiment_timeline': []
                    },
                    'move_timing': {
                        'start': move.get('move_start', idx).isoformat(),
                        'end': move.get('move_end', idx).isoformat(),
                        'window_start': window_start.isoformat(),
                        'window_end': window_end.isoformat()
                    }
                }
                
                if not window_sentiment.empty:
                    # Split sentiment before and after move
                    pre_move = window_sentiment[window_sentiment.index < idx]
                    post_move = window_sentiment[window_sentiment.index >= idx]
                    
                    self._debug_print(f"Pre-move records: {len(pre_move)}")
                    self._debug_print(f"Post-move records: {len(post_move)}")
                    
                    # Calculate sentiment metrics
                    if not pre_move.empty:
                        move_data['sentiment_data']['pre_move_sentiment'] = float(pre_move['avg_sentiment'].mean())
                        # Store detailed pre-move sentiment timeline
                        move_data['sentiment_data']['pre_move_timeline'] = [
                            {
                                'timestamp': ts.isoformat(),
                                'sentiment': float(val)
                            }
                            for ts, val in pre_move['avg_sentiment'].items()
                        ]
                        
                    if not post_move.empty:
                        move_data['sentiment_data']['post_move_sentiment'] = float(post_move['avg_sentiment'].mean())
                        # Store detailed post-move sentiment timeline
                        move_data['sentiment_data']['post_move_timeline'] = [
                            {
                                'timestamp': ts.isoformat(),
                                'sentiment': float(val)
                            }
                            for ts, val in post_move['avg_sentiment'].items()
                        ]
                    
                    if move_data['sentiment_data']['pre_move_sentiment'] is not None and \
                       move_data['sentiment_data']['post_move_sentiment'] is not None:
                        move_data['sentiment_data']['sentiment_shift'] = \
                            move_data['sentiment_data']['post_move_sentiment'] - \
                            move_data['sentiment_data']['pre_move_sentiment']
                    
                    move_data['sentiment_data']['news_volume'] = len(window_sentiment)
                    
                    self._debug_print("Move data:", move_data)
                
                correlations[timeframe]['moves'].append(move_data)
            
            # Calculate statistics for this timeframe
            moves_df = pd.DataFrame([m['sentiment_data'] for m in correlations[timeframe]['moves']])
            if not moves_df.empty:
                stats = {}
                try:
                    stats = {
                        'avg_move_size': float(moves_df['abs_change'].mean()) if 'abs_change' in moves_df else None,
                        'avg_pre_sentiment': float(moves_df['pre_move_sentiment'].mean()) if 'pre_move_sentiment' in moves_df else None,
                        'avg_post_sentiment': float(moves_df['post_move_sentiment'].mean()) if 'post_move_sentiment' in moves_df else None,
                        'avg_sentiment_shift': float(moves_df['sentiment_shift'].mean()) if 'sentiment_shift' in moves_df else None,
                        'avg_news_volume': float(moves_df['news_volume'].mean()) if 'news_volume' in moves_df else None,
                    }
                    
                    # Calculate correlation only if we have valid pre-move sentiment
                    if 'pre_move_sentiment' in moves_df and moves_df['pre_move_sentiment'].notna().any():
                        valid_data = moves_df.dropna(subset=['price_change', 'pre_move_sentiment'])
                        if not valid_data.empty:
                            stats['sentiment_move_correlation'] = float(valid_data['price_change'].corr(
                                valid_data['pre_move_sentiment']
                            ))
                        else:
                            stats['sentiment_move_correlation'] = None
                    else:
                        stats['sentiment_move_correlation'] = None
                        
                except Exception as e:
                    self._debug_print(f"Error calculating stats for {timeframe}: {str(e)}")
                    stats = {
                        'avg_move_size': None,
                        'avg_pre_sentiment': None,
                        'avg_post_sentiment': None,
                        'avg_sentiment_shift': None,
                        'avg_news_volume': None,
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
            
            if sentiment_metrics is None:
                sentiment_metrics = pd.DataFrame()
            
            # Enhanced summary structure
            summary = {
                'metadata': {
                    'symbol': self.symbol,
                    'analysis_period': {
                        'start': price_data.index[0].isoformat(),
                        'end': price_data.index[-1].isoformat()
                    },
                    'news_coverage': {
                        'total_news': len(analyzed_news),
                        'high_impact_news': len([n for n in analyzed_news if n.get('sentiment', {}).get('polarity', 0) > 0.7])
                    },
                    'timeframes_analyzed': self.timeframes
                },
                'price_moves': {},
                'sentiment_analysis': {}
            }
            
            self._debug_print("Analysis period:", summary['metadata']['analysis_period'])
            
            # Only proceed with correlation if we have sentiment metrics
            if not sentiment_metrics.empty:
                correlations = self._correlate_with_price_moves(price_data, sentiment_metrics)
                
                # Process each timeframe
                for tf in self.timeframes:
                    if tf in correlations:
                        moves_data = correlations[tf]
                        
                        summary['price_moves'][tf] = {
                            'total_moves': len(moves_data['moves']),
                            'significant_moves': [move for move in moves_data['moves'] 
                                               if abs(move['price_data']['price_change']) > 1.0],  # Significant threshold
                            'statistics': moves_data['stats'],
                            'moves_detail': moves_data['moves']
                        }
                        
                        # Add sentiment analysis summary, checking for column existence
                        summary['sentiment_analysis'][tf] = {
                            'average_sentiment': float(sentiment_metrics['avg_sentiment'].mean()) 
                                if 'avg_sentiment' in sentiment_metrics else 0.0,
                            'sentiment_volatility': float(sentiment_metrics['sentiment_volatility'].mean()) 
                                if 'sentiment_volatility' in sentiment_metrics else 0.0,
                            'total_news_events': int(sentiment_metrics['news_volume'].sum()) 
                                if 'news_volume' in sentiment_metrics else 0,
                            'correlation_stats': moves_data['stats'] if 'stats' in moves_data else {}
                        }
            else:
                print("No sentiment metrics available for correlation analysis")
                # Initialize empty structures
                for tf in self.timeframes:
                    summary['price_moves'][tf] = {
                        'total_moves': 0,
                        'significant_moves': [],
                        'statistics': {},
                        'moves_detail': []
                    }
                    summary['sentiment_analysis'][tf] = {
                        'average_sentiment': 0.0,
                        'sentiment_volatility': 0.0,
                        'total_news_events': 0,
                        'correlation_stats': {}
                    }
            
            return summary
            
        except Exception as e:
            print(f"Error in analyze_moves_and_sentiment: {str(e)}")
            raise