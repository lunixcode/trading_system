# Standard library imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
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
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.sentiment_history = pd.DataFrame()
        self.price_history = pd.DataFrame()
        self.significant_moves = pd.DataFrame()
        self.pattern_cache = {}
        self.timeframes = ['5min', '15min', '30min', '1H', '4H', '1D']

    def _calculate_price_moves(self, price_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate percentage price moves for given timeframe"""
        # Resample to desired timeframe
        resampled = price_data.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Calculate percentage moves
        resampled['price_change'] = resampled['Close'].pct_change() * 100
        resampled['high_low_range'] = ((resampled['High'] - resampled['Low']) / resampled['Low']) * 100
        resampled['move_type'] = resampled['price_change'].apply(lambda x: 'up' if x > 0 else 'down')
        
        return resampled

    def _find_significant_moves(self, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Find top 10 price moves for each timeframe"""
        significant_moves = {}
        
        for tf in self.timeframes:
            # Calculate moves for timeframe
            moves_df = self._calculate_price_moves(price_data, tf)
            
            # Find top 10 up and down moves
            top_up = moves_df[moves_df['price_change'] > 0].nlargest(10, 'price_change')
            top_down = moves_df[moves_df['price_change'] < 0].nsmallest(10, 'price_change')
            
            # Combine and sort by absolute value of price change
            top_moves = pd.concat([top_up, top_down])
            top_moves['abs_change'] = abs(top_moves['price_change'])
            top_moves = top_moves.sort_values('abs_change', ascending=False)
            
            # Add additional metrics
            top_moves['volume_change'] = top_moves['Volume'].pct_change()
            top_moves['volatility'] = (top_moves['High'] - top_moves['Low']) / top_moves['Open']
            
            significant_moves[tf] = top_moves

        return significant_moves

    def _calculate_sentiment_metrics(self, analyzed_news: List[Dict]) -> pd.DataFrame:
        """Calculate sentiment metrics for different timeframes"""
        if not analyzed_news:
            return pd.DataFrame()
            
        # Convert news data to DataFrame
        news_df = pd.DataFrame(analyzed_news)
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
        news_df.set_index('timestamp', inplace=True)
        
        metrics_list = []
        for tf in self.timeframes:
            # Resample news data to timeframe
            grouped = news_df.resample(tf)
            
            # Calculate metrics for each period
            for name, group in grouped:
                if not group.empty:
                    metrics = SentimentMetrics(
                        period_start=group.index[0],
                        period_end=group.index[-1],
                        avg_sentiment=group['sentiment_score'].mean(),
                        news_volume=len(group),
                        significant_news=len(group[group['impact_score'] >= 7]),
                        sentiment_momentum=group['sentiment_score'].diff().mean(),
                        dominant_category=group['category'].mode()[0],
                        volatility=group['sentiment_score'].std()
                    )
                    metrics_list.append(vars(metrics))
        
        return pd.DataFrame(metrics_list)

    def _correlate_with_price_moves(self, 
                                  price_data: pd.DataFrame, 
                                  sentiment_metrics: pd.DataFrame) -> Dict:
        """Find correlations between sentiment and significant price movements"""
        # Find significant moves first
        significant_moves = self._find_significant_moves(price_data)
        correlations = {}
        
        for timeframe, moves in significant_moves.items():
            correlations[timeframe] = {
                'moves': [],
                'sentiment_context': [],
                'stats': {}
            }
            
            for idx, move in moves.iterrows():
                # Look for news/sentiment in window around the move
                window_start = idx - pd.Timedelta(hours=24)  # Look back 24h
                window_end = idx + pd.Timedelta(hours=24)    # Look forward 24h
                
                # Find sentiment metrics in this window
                window_sentiment = sentiment_metrics[
                    (sentiment_metrics.index >= window_start) & 
                    (sentiment_metrics.index <= window_end)
                ]
                
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
                    
                    # Calculate sentiment metrics
                    if not pre_move.empty:
                        move_data['pre_move_sentiment'] = pre_move['avg_sentiment'].mean()
                    if not post_move.empty:
                        move_data['post_move_sentiment'] = post_move['avg_sentiment'].mean()
                    
                    if move_data['pre_move_sentiment'] and move_data['post_move_sentiment']:
                        move_data['sentiment_shift'] = move_data['post_move_sentiment'] - move_data['pre_move_sentiment']
                    
                    move_data['news_volume'] = len(window_sentiment)
                
                correlations[timeframe]['moves'].append(move_data)
            
            # Calculate statistics for this timeframe
            moves_df = pd.DataFrame(correlations[timeframe]['moves'])
            if not moves_df.empty:
                correlations[timeframe]['stats'] = {
                    'avg_move_size': moves_df['price_change'].abs().mean(),
                    'avg_pre_sentiment': moves_df['pre_move_sentiment'].mean(),
                    'avg_post_sentiment': moves_df['post_move_sentiment'].mean(),
                    'avg_sentiment_shift': moves_df['sentiment_shift'].mean(),
                    'avg_news_volume': moves_df['news_volume'].mean(),
                    'sentiment_move_correlation': moves_df['price_change'].corr(
                        moves_df['pre_move_sentiment']
                    ) if 'pre_move_sentiment' in moves_df else None
                }
        
        return correlations

    async def analyze_moves_and_sentiment(self, price_data: pd.DataFrame, analyzed_news: List[Dict]) -> Dict:
        """Main method to analyze price moves and correlate with sentiment"""
        try:
            # Store price data
            self.price_history = price_data
            
            # Calculate sentiment metrics
            sentiment_metrics = self._calculate_sentiment_metrics(analyzed_news)
            
            # Find correlations
            correlations = self._correlate_with_price_moves(price_data, sentiment_metrics)
            
            # Prepare summary
            summary = {
                'analysis_period': {
                    'start': price_data.index[0],
                    'end': price_data.index[-1]
                },
                'price_moves': {},
                'sentiment_correlations': correlations,
                'news_coverage': {
                    'total_news': len(analyzed_news),
                    'high_impact_news': len([n for n in analyzed_news if n.get('impact_score', 0) >= 7])
                }
            }
            
            # Add summary stats for each timeframe
            for tf in self.timeframes:
                if tf in correlations:
                    summary['price_moves'][tf] = {
                        'total_significant_moves': len(correlations[tf]['moves']),
                        'stats': correlations[tf]['stats']
                    }
            
            return summary
            
        except Exception as e:
            print(f"Error in analyze_moves_and_sentiment: {str(e)}")
            return {}