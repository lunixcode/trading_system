import requests
import json
import asyncio
from datetime import datetime
from difflib import SequenceMatcher
from dotenv import load_dotenv
import os

# Load the environment variables from .env
load_dotenv()

# Get the API keys
STRAWBERRY_API_KEY  = os.getenv('STRAWBERRY_API_KEY ')
EOD_API_KEY = os.getenv('EOD_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

NEWS_URL = "https://eodhistoricaldata.com/api/news"
OUTPUT_FILE = "articles.json"
COMPANIES = ['NVDA.US']

print(f"Loaded API key: {'Key exists' if OPENAI_API_KEY else 'Key not found'}")
print(f"API key value: {OPENAI_API_KEY[:5]}..." if OPENAI_API_KEY else "No key loaded")


# Headers for OpenAI requests
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Store processed articles and high-impact articles
processed_articles = set()
high_impact_articles = []

def fetch_latest_news(tickers):
    """Fetches the latest news for the given tickers."""
    tickers_str = ','.join(tickers)
    params = {
        'api_token': EOD_API_KEY,
        's': tickers_str,
        'limit': 10,
        'fmt': 'json'
    }
    try:
        response = requests.get(NEWS_URL, params=params)
        if response.status_code != 200:
            print(f"Error fetching news for {tickers}: Status code {response.status_code}")
            return []
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []

async def call_gpt_4_turbo(prompt):
    """Send a prompt to the GPT-4 Turbo model."""
    data = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.5
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=HEADERS,
            data=json.dumps(data)
        )
        response_json = response.json()
        if response.status_code == 200:
            return response_json['choices'][0]['message']['content']
        else:
            print(f"Error with GPT-4 Turbo API: {response_json}")
            return "Unable to process with GPT-4 Turbo"
    except Exception as e:
        print(f"Exception calling GPT-4 Turbo: {e}")
        return "Error with GPT-4 Turbo"

async def call_detailed_analysis(content, combined_summary):
    """Use GPT-4 Turbo for deeper analysis."""
    prompt = f"""
    You are a leading quantative analysis at goldman sachs, i need you to perform a detailed analysis on the following combined news content:
    - Assess sentiment (0-10).
    - Asses relevancy (0-10)
    - Evaluate reliability of source (0-10).
    - Provide risk assessment (0-10).
    - Assign an impact score between (0-10).
    - Suggest a short-term investment strategy (hours to days).
    - If the combined values of the of the top 5 categories equal or above 40 create an action plan for immediate action with a review period (Buy, Sell, Hold etc)
    
    Combined content:
    {combined_summary}

    New update:
    {content}
    """
    data = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,  # Adjust based on your needs
        "temperature": 0.5
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=HEADERS,
            data=json.dumps(data)
        )
        response_json = response.json()
        if response.status_code == 200:
            return response_json['choices'][0]['message']['content']
        else:
            print(f"Error querying GPT-4 Turbo: {response_json}")
            return "Unable to perform advanced analysis with GPT-4 Turbo"
    except Exception as e:
        print(f"Exception calling GPT-4 Turbo: {e}")
        return "Error with GPT-4 Turbo API"


async def categorize_summarize_and_score(article_text, source, date):
    """Use GPT-4 Turbo to categorize, summarize, and score the article."""
    prompt = f"""
    Please perform the following tasks:

    Primary check: 
        - The news must relate to Nvidia stocks otherwise exit response here with non-related tag but print out what you have if nots related

    1. Categorize the news article based on these categories:
       - Official Quarterly Numbers (only from trusted sources).
       - Unofficial Earnings Estimates.
       - Official Regulatory Updates.
       - Speculative Regulatory Updates.
       - Macroeconomic Impact.
       - Product Announcements.
       - Strategic Changes.
       - Market Sentiment.
       - Analyst Upgrades/Downgrades.
       - Insider Buying/Selling.
       - Stock Splits or Buybacks.
       - Mergers, Acquisitions, or Divestitures.
       - Competitor Performance.
       - Market Share Shifts.
       - Technological Developments.
       - Advances in R&D.
       - Patent Approvals or Disputes.
       - Cybersecurity Incidents.
       - Supply Chain Disruptions.
       - Manufacturing Changes.
       - Environmental, Social, and Governance (ESG) Updates.
       - Sustainability Initiatives.
       - Corporate Social Responsibility Efforts.
       - Geopolitical Events.
       - Trade Wars or Tariff Changes.
       - Political Elections and Their Market Impact.
       - Wars, Sanctions, or Military Actions.
       - General Market Trends.
       - Retail Investor Behavior.
       - Energy and Commodities Updates.
       - Cryptocurrency and Blockchain Developments.
       - Media and Entertainment.
       - Retail and E-commerce.
       - Speculative or Rumor-based News.
       - Unexpected Events or Scandals.
       - Other.

    2. Check if it is one of these
        - Official Quarterly Numbers (only from trusted sources).
        - Unofficial Earnings Estimates.
        - Official Regulatory Updates.
        - Speculative Regulatory Updates.
        - Macroeconomic Impact.
        - Product Announcements.
        - Strategic Changes.
        - Market Sentiment.
        - Other."

    3. Provide a brief summary.
       
    4. Assign an impact score between 1-10. 
       - High impact means there will be a noticable response in the market
       - Use an integer between 1 and 10,
       - ONLY scores >5 indicate HIGH IMPACT, 
       - There MUST be a market reaction if the score is over 7. 
       - The response should be formatted "**Impact**: valueHere"
    
    Article Source: {source}
    Article Date: {date}
    Article:
    {article_text}
    """
    
    result = await call_gpt_4_turbo(prompt)
    print(f"\nGPT-4 Response: {result}")

    result = await call_gpt_4_turbo(prompt) 
    print(f"\nGPT-4 Response: {result}")

# Attempt to extract the impact score more robustly
    if "**Impact**:" in result:
        try:
            # Split by the impact label and extract the score
            score_text = result.split("**Impact**:")[1].strip()
            # Handle cases with additional text after the number
            score = int(score_text.split()[0])  # Extract the first number from the split
            print(f"Extracted Impact Score: {score}")
            return score, result
        except (ValueError, IndexError) as e:
            # Handle errors in extraction
            print(f"Error extracting impact score: {e}")
            return 0, result
    else:
        print("Impact score not found in GPT-4 response.")
        return 0, result


async def process_article(article):
    """Process each article."""
    title = article.get('title', 'No Title')
    content = article.get('content', 'No Content Available')
    source = article.get('source', 'Unknown Source')
    date = article.get('date', 'Unknown Date')

    article_id = f"{title}-{date}-{source}"
    if article_id in processed_articles:
        return

    processed_articles.add(article_id)

    if content:
        score, result = await categorize_summarize_and_score(content, source, date)
        print(f"Debug: the score was : {score} \n")
        if score >= 6:
            print("High impact detected, calling Strawberry for deeper analysis...")
            combined_summary = content
            detailed_analysis = await call_detailed_analysis(content, combined_summary)
            print(f"\nStrawberry Analysis:\n{detailed_analysis}")

async def monitor_news():
    """Continuously fetch and process news articles."""
    while True:
        news_data = fetch_latest_news(COMPANIES)
        if news_data:
            tasks = [process_article(article) for article in news_data]
            await asyncio.gather(*tasks)
        
        print("Waiting for 15 minutes before the next request...")
        await asyncio.sleep(900)

if __name__ == "__main__":
    asyncio.run(monitor_news())
