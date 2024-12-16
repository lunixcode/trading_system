import asyncio
import json
import requests
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging
from dotenv import load_dotenv
import os

class newsAnalysisNew:
    """Analyzes news articles using two-stage AI processing"""
    
    def __init__(self, symbol: str):
        self._setup_environment()
        self._setup_logging()
        self.symbol = symbol
        self.processed_articles = set()
        
    def _setup_environment(self):
        """Initialize API keys and configurations"""
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initial_analysis(self, news_item: Dict) -> Tuple[int, str]:
        """First stage analysis for category and impact"""
        
        prompt = f"""
        Please perform the following tasks:

        Primary check: 
            - The news must relate to {self.symbol} stocks otherwise exit response here with non-related tag

        1. Categorize the news article based on these categories:
           - Official Quarterly Numbers
           - Unofficial Earnings Estimates
           - Official Regulatory Updates
           - Product Announcements
           - Strategic Changes
           - Market Sentiment
           - Analyst Upgrades/Downgrades
           - Insider Trading
           - Other

        2. Provide a brief summary.
           
        3. Assign an impact score between 1-10. 
           - High impact means there will be a noticeable response in the market
           - Use an integer between 1 and 10
           - ONLY scores >5 indicate HIGH IMPACT
           - There MUST be a market reaction if the score is over 7
           - The response should be formatted "**Impact**: valueHere"
        
        Article Source: {news_item.get('source', 'Unknown')}
        Article Date: {news_item.get('date', 'Unknown')}
        Article:
        {news_item.get('content', '')}
        """
        
        try:
            result = await self._call_gpt(prompt)
            impact_score = self._extract_impact_score(result)
            return impact_score, result
        except Exception as e:
            self.logger.error(f"Error in initial analysis: {e}")
            return 0, str(e)

    async def detailed_analysis(self, news_item: Dict, initial_analysis: str) -> Dict:
        """Second stage detailed analysis for high-impact news"""
        
        prompt = f"""
        You are a leading quantitative analyst at Goldman Sachs. Perform a detailed analysis on this news:
        
        1. Score these metrics (0-10):
           - Sentiment
           - Relevancy
           - Source Reliability
           - Risk Level
           - Market Impact
        
        2. Based on the above scores:
           - If total score ≥ 40: Provide specific trading action plan
           - Include position sizing recommendation
           - Set review period (hours/days)
           - Specify exit conditions
        
        Initial Analysis:
        {initial_analysis}
        
        New Content:
        {news_item.get('content', '')}
        """
        
        try:
            result = await self._call_gpt(prompt, max_tokens=4096)
            return self._parse_detailed_analysis(result)
        except Exception as e:
            self.logger.error(f"Error in detailed analysis: {e}")
            return {"error": str(e)}

    async def _call_gpt(self, prompt: str, max_tokens: int = 300) -> str:
        """Make API call to GPT-4"""
        data = {
            "model": "gpt-4-turbo-preview",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.5
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f"GPT API call failed: {e}")
            raise

    def _extract_impact_score(self, result: str) -> int:
        """Extract impact score from GPT response"""
        try:
            if "**Impact**:" in result:
                score_text = result.split("**Impact**:")[1].strip()
                return int(score_text.split()[0])
            return 0
        except Exception as e:
            self.logger.error(f"Error extracting impact score: {e}")
            return 0

    def _parse_detailed_analysis(self, result: str) -> Dict:
        """Parse the detailed analysis response into structured data"""
        try:
            # Basic structure for parsed results
            parsed = {
                "scores": {
                    "sentiment": 0,
                    "relevancy": 0,
                    "reliability": 0,
                    "risk": 0,
                    "impact": 0
                },
                "action_plan": None,
                "position_size": None,
                "review_period": None,
                "exit_conditions": None,
                "raw_analysis": result
            }
            
            # Add more sophisticated parsing logic here
            # This is a placeholder for the actual parsing implementation
            
            return parsed
        except Exception as e:
            self.logger.error(f"Error parsing detailed analysis: {e}")
            return {"error": str(e), "raw_analysis": result}

    async def analyze_news(self, news_item: Dict) -> Dict:
        """Main method to analyze a news item"""
        
        # Create unique ID for the article
        article_id = f"{news_item.get('title', '')}-{news_item.get('date', '')}"
        
        # Skip if already processed
        if article_id in self.processed_articles:
            return {"status": "already_processed"}
        
        self.processed_articles.add(article_id)
        
        # Perform initial analysis
        impact_score, initial_result = await self.initial_analysis(news_item)
        
        analysis_result = {
            "initial_analysis": initial_result,
            "impact_score": impact_score,
            "detailed_analysis": None
        }
        
        # If high impact, perform detailed analysis
        if impact_score >= 6:
            detailed_result = await self.detailed_analysis(news_item, initial_result)
            analysis_result["detailed_analysis"] = detailed_result
        
        return analysis_result

# Example usage in backtester:
"""
# In your backtesting strategy
class NewsStrategy(bt.Strategy):
    def __init__(self):
        self.news_analyzer = NewsAnalyzer(symbol="NVDA")
        
    async def analyze_news_impact(self, news_item):
        result = await self.news_analyzer.analyze_news(news_item)
        return result
        
    def next(self):
        if self.data.news_count[0] > self.data.news_count[-1]:
            # Get news for current minute
            current_news = self.get_current_news()  # You'll need to implement this
            
            # Analyze the news
            analysis = asyncio.run(self.analyze_news_impact(current_news))
            
            # Make trading decisions based on analysis
            if analysis['impact_score'] >= 7:
                if analysis['detailed_analysis']:
                    # Implement trading logic based on detailed analysis
                    pass
"""