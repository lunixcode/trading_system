import os
import requests
import time
import logging
from datetime import datetime
from typing import List, Dict

class WebSearchNewsFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Web Search endpoint (not the News endpoint)
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"

        # Only three companies
        self.companies = {
            "NVDA": "Nvidia",
            "MSFT": "Microsoft",
            "GOOGL": "Alphabet"
        }

        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def fetch_company_news(self, company_name: str, ticker: str) -> List[Dict]:
        """
        Use Bing Web Search endpoint with 'responseFilter=News' to get news-like results.
        """
        try:
            params = {
                "q": f"{company_name} OR {ticker} stock"
                #"responseFilter": "News",   # Important: filter out everything but news
                #'"count": 10                # number of results
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Bing Web Search with responseFilter=News typically puts articles under data["news"]["value"]
            news_items = data.get("news", {}).get("value", [])

            processed_news = []
            for article in news_items:
                processed_news.append({
                    "company": company_name,
                    "ticker": ticker,
                    "title": article.get("name"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "published": article.get("datePublished"),
                    "source": (article.get("provider", [{}])[0].get("name")
                               if article.get("provider") else None),
                    "timestamp": datetime.now().isoformat()
                })
            
            return processed_news

        except requests.RequestException as e:
            self.logger.error(f"Error fetching news for {company_name}: {str(e)}")
            return []

    def fetch_all_news(self) -> List[Dict]:
        """
        Fetch news-like results for all three companies.
        """
        all_news = []
        for ticker, company in self.companies.items():
            self.logger.info(f"Fetching (web-based) news for {company} ({ticker})")
            company_news = self.fetch_company_news(company, ticker)
            all_news.extend(company_news)
            
            # Sleep a bit between calls if needed
            time.sleep(2)
        
        return all_news

    def start_monitoring(self, callback=None):
        """
        Continuously fetch web-based 'news' every minute.
        """
        while True:
            try:
                self.logger.info("Fetching news updates...")
                news = self.fetch_all_news()
                
                if callback and news:
                    callback(news)
                else:
                    self.logger.info(f"Found {len(news)} news articles (via Web Search)")
                
                time.sleep(60)  # Wait 1 minute before the next round

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)

# Example usage:
if __name__ == "__main__":
    API_KEY = os.getenv('MICROSOFT_API_KEY')  # or paste your key here for testing

    fetcher = WebSearchNewsFetcher(API_KEY)

    # Simple callback to print new articles
    def handle_news(news_items):
        for item in news_items:
            print(f"{item['company']} (Ticker: {item['ticker']}) - {item['title']}")

    fetcher.start_monitoring(callback=handle_news)
