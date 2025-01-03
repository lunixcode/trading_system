import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
import logging

class NewsFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.bing.microsoft.com/v7.0/news/search"
        self.companies = {
            "NVDA": "Nvidia",
            "TSLA": "Tesla",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "META": "Meta"
        }
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
        }
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fetch_company_news(self, company_name: str, ticker: str) -> List[Dict]:
        """
        Fetch news for a specific company
        """
        try:
            params = {
                "q": f"{company_name} OR {ticker} stock",
                "count": 10,  # Number of news articles per company
                "freshness": "Day",  # Get news from the last 24 hours
                "sortBy": "Date"  # Get most recent news first
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            
            news_data = response.json()
            
            # Process and clean the news data
            processed_news = []
            for article in news_data.get("value", []):
                processed_news.append({
                    "company": company_name,
                    "ticker": ticker,
                    "title": article.get("name"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "published": article.get("datePublished"),
                    "source": article.get("provider", [{}])[0].get("name"),
                    "timestamp": datetime.now().isoformat()
                })
            
            return processed_news
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching news for {company_name}: {str(e)}")
            return []

    def fetch_all_news(self) -> List[Dict]:
        """
        Fetch news for all companies
        """
        all_news = []
        for ticker, company in self.companies.items():
            self.logger.info(f"Fetching news for {company} ({ticker})")
            company_news = self.fetch_company_news(company, ticker)
            all_news.extend(company_news)
            time.sleep(2)  # Rate limiting between requests
        
        return all_news

    def start_monitoring(self, callback=None):
        """
        Start continuous monitoring with 1-minute intervals
        callback: Optional function to handle the news data
        """
        while True:
            try:
                self.logger.info("Fetching news updates...")
                news = self.fetch_all_news()
                
                if callback and news:
                    callback(news)
                else:
                    self.logger.info(f"Found {len(news)} news articles")
                
                time.sleep(60)  # Wait for 1 minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait before retrying

# Example usage:
if __name__ == "__main__":
    API_KEY = "your_bing_api_key"
    
    # Initialize the news fetcher
    news_fetcher = NewsFetcher(API_KEY)
    
    # Example callback function
    def handle_news(news_items):
        for item in news_items:
            print(f"New article for {item['company']}: {item['title']}")
    
    # Start monitoring with the callback
    news_fetcher.start_monitoring(callback=handle_news)