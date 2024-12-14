from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

class DataValidator:
    """
    Validates data integrity and quality for historical financial data.
    Works in conjunction with HistoricalDataManager to ensure data reliability.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        if self.debug:
            print("Initializing DataValidator")

    def validate_price_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates price data for common issues and integrity.
        
        Args:
            df: DataFrame containing price data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if self.debug:
            print("\n=== Validating Price Data ===")
            print(f"Input DataFrame shape: {df.shape}")
            print(f"Columns present: {df.columns.tolist()}")
        
        issues = []
        
        # Required columns check
        required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            if self.debug:
                print(f"❌ Missing columns detected: {missing_columns}")
            issues.append(f"Missing required columns: {missing_columns}")
        else:
            if self.debug:
                print("✓ All required columns present")
        
        if not issues:  # Only continue if we have all required columns
            # Check for missing values
            null_counts = df[list(required_columns)].isnull().sum()
            if null_counts.any():
                if self.debug:
                    print("❌ Missing values detected:")
                    for col, count in null_counts[null_counts > 0].items():
                        print(f"   - {col}: {count} missing values")
                issues.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")
            else:
                if self.debug:
                    print("✓ No missing values found")
            
            # Check for duplicate dates
            duplicate_dates = df['Date'].duplicated()
            if duplicate_dates.any():
                dup_dates = df['Date'][duplicate_dates].tolist()
                if self.debug:
                    print(f"❌ Found {len(dup_dates)} duplicate dates")
                issues.append(f"Duplicate dates found: {dup_dates}")
            else:
                if self.debug:
                    print("✓ No duplicate dates found")
            
            # Validate price relationships
            invalid_prices = (
                (df['High'] < df['Low']) |
                (df['Open'] > df['High']) |
                (df['Open'] < df['Low']) |
                (df['Close'] > df['High']) |
                (df['Close'] < df['Low'])
            )
            if invalid_prices.any():
                invalid_dates = df['Date'][invalid_prices].tolist()
                if self.debug:
                    print(f"❌ Found {sum(invalid_prices)} invalid price relationships")
                    print("Examples of invalid price relationships:")
                    print(df[invalid_prices].head())
                issues.append(f"Invalid price relationships found on dates: {invalid_dates}")
            else:
                if self.debug:
                    print("✓ All price relationships valid")
            
            # Check for negative values
            negative_values = (df[['Open', 'High', 'Low', 'Close', 'Volume']] < 0).any()
            if negative_values.any():
                neg_cols = negative_values[negative_values].index.tolist()
                if self.debug:
                    print(f"❌ Found negative values in columns: {neg_cols}")
                issues.append(f"Negative values found in columns: {neg_cols}")
            else:
                if self.debug:
                    print("✓ No negative values found")
            
            # Check for chronological order
            if not df['Date'].is_monotonic_increasing:
                if self.debug:
                    print("❌ Dates are not in chronological order")
                    print("First few dates:")
                    print(df['Date'].head())
                    print("Last few dates:")
                    print(df['Date'].tail())
                issues.append("Dates are not in chronological order")
            else:
                if self.debug:
                    print("✓ Dates are in chronological order")

            # Additional statistics in debug mode
            if self.debug:
                print("\nData Statistics:")
                print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
                print("\nPrice Ranges:")
                for col in ['Open', 'High', 'Low', 'Close']:
                    print(f"{col}: {df[col].min():.2f} to {df[col].max():.2f}")
                print(f"Volume Range: {df['Volume'].min()} to {df['Volume'].max()}")
        
        is_valid = len(issues) == 0
        if self.debug:
            print(f"\nValidation {'Passed' if is_valid else 'Failed'}")
            if not is_valid:
                print("Issues found:")
                for issue in issues:
                    print(f"- {issue}")
                    
        return is_valid, issues

    def validate_news_data(self, news_items: List[dict]) -> Tuple[bool, List[str], List[dict]]:
        """
        Validates news data and filters out items with 00:00:00 times.
        
        Args:
            news_items: List of news item dictionaries
            
        Returns:
            Tuple of (is_valid, list_of_issues, filtered_news_items)
        """
        if self.debug:
            print("\n=== Validating News Data ===")
            print(f"Number of news items to validate: {len(news_items)}")
        
        issues = []
        valid_news_items = []
        required_fields = {'date', 'title', 'content', 'link'}
        
        for idx, item in enumerate(news_items):
            if self.debug and idx % 100 == 0:  # Print progress every 100 items
                print(f"Processing item {idx + 1}/{len(news_items)}")
            
            item_valid = True
            
            # Check required fields
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                if self.debug:
                    print(f"Item {idx}: Missing fields: {missing_fields}")
                issues.append(f"Item {idx}: Missing required fields: {missing_fields}")
                item_valid = False
                continue
            
            # Validate date and time
            try:
                news_date = datetime.fromisoformat(item['date'])
                
                # Skip items with 00:00:00 time
                if news_date.hour == 0 and news_date.minute == 0 and news_date.second == 0:
                    #if self.debug:
                        #print(f"Item {idx}: Skipping due to zero time: {item['date']}")
                    continue
                
            except ValueError as e:
                if self.debug:
                    print(f"Item {idx}: Invalid date format: {item['date']}")
                issues.append(f"Item {idx}: Invalid date format: {item['date']}")
                item_valid = False
                continue
            
            # Check for empty content
            if not item['content'].strip():
                if self.debug:
                    print(f"Item {idx}: Empty content")
                issues.append(f"Item {idx}: Empty content")
                item_valid = False
                continue
            
            # Check for empty title
            if not item['title'].strip():
                if self.debug:
                    print(f"Item {idx}: Empty title")
                issues.append(f"Item {idx}: Empty title")
                item_valid = False
                continue
            
            # If all checks pass, add to valid items
            if item_valid:
                valid_news_items.append(item)
        
        if self.debug:
            print("\nValidation Summary:")
            print(f"Original items: {len(news_items)}")
            print(f"Valid items with non-zero times: {len(valid_news_items)}")
            print(f"Items removed: {len(news_items) - len(valid_news_items)}")
            if valid_news_items:
                print("\nSample of valid items:")
                for item in valid_news_items[:3]:
                    print(f"Date: {item['date']}")
        
        is_valid = len(issues) == 0 and len(valid_news_items) > 0
        
        return is_valid, issues, valid_news_items

    def validate_all_data(self, price_data: pd.DataFrame, news_data: List[dict]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validates all data types and their relationships.
        
        Args:
            price_data: DataFrame containing price data
            news_data: List of news items
            
        Returns:
            Tuple of (is_valid, dict_of_issues_by_type)
        """
        if self.debug:
            print("\n=== Starting Complete Data Validation ===")
        
        all_issues = {}
        
        # Validate price data
        price_valid, price_issues = self.validate_price_data(price_data)
        if not price_valid:
            all_issues['price'] = price_issues
            
        # Validate news data
        news_valid, news_issues = self.validate_news_data(news_data)
        if not news_valid:
            all_issues['news'] = news_issues
            
        # Cross-validation between different data types
        if price_valid and news_valid:
            if self.debug:
                print("\nPerforming cross-validation between price and news data")
            
            price_start = price_data['Date'].min()
            price_end = price_data['Date'].max()
            
            news_alignment_issues = []
            for idx, news_item in enumerate(news_data):
                news_date = datetime.fromisoformat(news_item['date'])
                if news_date.date() < price_start.date() or news_date.date() > price_end.date():
                    if self.debug:
                        print(f"❌ News item {idx} date {news_date.date()} outside price data range")
                    news_alignment_issues.append(
                        f"News item date {news_date.date()} outside of price data range ({price_start.date()} to {price_end.date()})"
                    )
            
            if news_alignment_issues:
                all_issues['alignment'] = news_alignment_issues
        
        is_valid = len(all_issues) == 0
        if self.debug:
            print(f"\nComplete Validation {'Passed' if is_valid else 'Failed'}")
            if not is_valid:
                print("Issues found by category:")
                for category, issues in all_issues.items():
                    print(f"\n{category.upper()} ISSUES:")
                    for issue in issues:
                        print(f"- {issue}")
        
        return is_valid, all_issues