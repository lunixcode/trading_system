import google.generativeai as genai
import os
import asyncio
import requests
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EOD_API_KEY = os.getenv("EOD_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in your environment variables.")
if not EOD_API_KEY:
    raise ValueError("EOD_API_KEY is not set in your environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# Define models
STEP1_MODEL = genai.GenerativeModel("gemini-1.5-pro")
STEP2_MODEL = genai.GenerativeModel("gemini-2.0-flash-exp")

NEWS_URL = "https://eodhistoricaldata.com/api/news"
COMPANIES = ["NVDA.US"]



# Define the responses directory
RESPONSES_DIR = Path(__file__).parent / "responses"

# Ensure the 'responses' folder exists
RESPONSES_DIR.mkdir(exist_ok=True)  # This creates the folder if it doesn't exist.

async def call_gemini(model, prompt, delay=1):
    """
    Call the Gemini LLM using the generate_content_async method.
    """
    try:
        response = await model.generate_content_async(prompt)
        await asyncio.sleep(delay)  # Add delay to respect API limits
        print(f"Response from {model.model_name}: {response.text}")
        return response.text
    except Exception as e:
        print(f"Error calling Gemini model {model.model_name}: {e}")
        return f"Error: {e}"

def fetch_latest_news(tickers):
    """
    Fetch the latest news for the given tickers from EOD API.
    """
    tickers_str = ",".join(tickers)
    params = {
        "api_token": EOD_API_KEY,
        "s": tickers_str,
        "limit": 10,
        "fmt": "json"
    }
    try:
        response = requests.get(NEWS_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

async def save_response(article, step1_result, detailed_result=None):
    """
    Save the analysis results to a JSON file.
    """
    # Generate a timestamp for the file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = RESPONSES_DIR / f"response_{timestamp}.json"

    # Prepare the data to save
    data = {
        "article": article,
        "step1_result": step1_result,
        "detailed_result": detailed_result
    }

    # Save the data to the JSON file
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)

async def categorize_summarize_and_score(article_text, source, date):
    """
    Perform Step 1 analysis using the Gemini Step 1 model.
    """
    prompt = f"""
    Please perform the following tasks:

    Primary check: 
        - The news must relate to Nvidia stocks (NVDA) otherwise exit response here
          with non-related tag but do provide a brief mention of relevance.

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

    2. Check if it is one of these (choose only one):
       - Official Quarterly Numbers (only from trusted sources).
       - Unofficial Earnings Estimates.
       - Official Regulatory Updates.
       - Speculative Regulatory Updates.
       - Macroeconomic Impact.
       - Product Announcements.
       - Strategic Changes.
       - Market Sentiment.
       - Other.

    3. Provide a brief summary.

    4. Assign an impact score between 1-10:
       - High impact means there will be a noticeable response in the market
       - ONLY scores >5 indicate HIGH IMPACT
       - Scores >7 must imply a strong market reaction.
       - Format the response with "**Impact**: valueHere"

    Article Source: {source}
    Article Date: {date}
    Article:
    {article_text}
    """
    result = await call_gemini(STEP1_MODEL, prompt)
    print(f"Step 1 Response: {result}")

    # Extract the impact score
    impact_score = 0
    if "**Impact**:" in result:
        try:
            impact_score = int(result.split("**Impact**:")[1].strip().split()[0])
        except (ValueError, IndexError):
            print("Failed to extract impact score.")

    return impact_score, result

async def detailed_analysis(content, combined_summary):
    """
    Perform detailed analysis using the Step 2 Gemini model.
    """
    prompt = f"""
    Perform a detailed analysis of the following content as a hedge-fund trader:

    1. Sentiment (0-10) and a brief reason.
    2. Relevance (0-10) and a brief reason.
    3. Source reliability (0-10) and a brief reason.
    4. Risk (0-10) and a brief reason.
    5. Market impact (0-10) and a brief reason.
    6. Volatility outlook (0-10) and a brief reason.
    7. Recommended strategy (Buy, Sell, Hold) with timeframe.
    8. Recommended stop-loss and take-profit levels.

    Combined Summary:
    {combined_summary}

    New Content:
    {content}
    """
    result = await call_gemini(STEP2_MODEL, prompt)
    print(f"Step 2 Response: {result}")
    return result

async def process_article(article):
    """
    Process an article through Step 1 and optionally Step 2 analysis.
    """
    title = article.get("title", "No Title")
    content = article.get("content", "No Content Available")
    source = article.get("source", "Unknown Source")
    date = article.get("date", "Unknown Date")

    print(f"Processing article: {title}")

    impact_score, step1_result = await categorize_summarize_and_score(content, source, date)
    detailed_result = None

    if impact_score >= 6:
        print(f"High impact detected for {title}. Proceeding to Step 2 analysis.")
        detailed_result = await detailed_analysis(content, step1_result)
        print(f"Detailed Analysis Result: {detailed_result}")
    else:
        print(f"Low impact for {title}. Skipping detailed analysis.")

    await save_response(article, step1_result, detailed_result)
    await asyncio.sleep(1)  # Ensure 1 second delay between processing

async def main():
    while True:
        news_data = fetch_latest_news(COMPANIES)
        if news_data:
            tasks = [process_article(article) for article in news_data]
            await asyncio.gather(*tasks)
        print("Waiting 15 minutes for the next news fetch...")
        await asyncio.sleep(900)

if __name__ == "__main__":
    asyncio.run(main())
