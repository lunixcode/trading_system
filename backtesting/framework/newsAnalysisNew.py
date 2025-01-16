# Standard library imports
import os
import asyncio
import json
from typing import Dict, Tuple, Optional, Literal

# Third-party imports
import requests
import google.generativeai as genai
import anthropic
import together
from dotenv import load_dotenv

class NewsAnalysis:
    """Analyzes news articles using multiple AI models with two-stage processing"""
    
    def __init__(self, symbol: str, model_choice: Literal["gpt", "gemini", "claude", "together"] = "gpt"):
        self._setup_environment()
        self.symbol = symbol
        self.model_choice = model_choice
        self.processed_articles = set()
        
    def _setup_environment(self):
        """Initialize API keys and configurations"""
        load_dotenv()
        
        # OpenAI setup
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            self.openai_headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
        # Gemini setup
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            
        # Claude setup
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY')
        if self.claude_api_key:
            self.claude_client = anthropic.Anthropic(
                api_key=self.claude_api_key
            )
            
        # Together AI setup
        self.together_api_key = os.getenv('TOGETHER_API_KEY')
        if self.together_api_key:
            together.api_key = self.together_api_key

    async def _call_ai_model(self, prompt: str, detailed: bool = False) -> str:
        """Make API call to selected AI model"""
        try:
            print(f"\n{'='*50}")
            print(f"LLM API Call:")
            print(f"Model Choice: {self.model_choice}")
            print(f"Detailed Mode: {detailed}")

            if self.model_choice == "gpt":
                model = "gpt-4-turbo-preview" if detailed else "gpt-3.5-turbo"
                print(f"Using GPT Model: {model}")
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096 if detailed else 500,
                    "temperature": 0.5
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.openai_headers,
                    json=data
                )
                
                if response.status_code == 429:
                    print("\nGPT: Hit rate limit, waiting 20 seconds...")
                    await asyncio.sleep(20)
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=self.openai_headers,
                        json=data
                    )
                
                await asyncio.sleep(0.5 if model == "gpt-4-turbo-preview" else 0.2)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
                    
            elif self.model_choice == "claude":
                model = "claude-3-opus-20240229" if detailed else "claude-3-sonnet-20240229"
                print(f"Using Claude Model: {model}")
                try:
                    print("Sending request to Claude API...")
                    response = self.claude_client.messages.create(
                        model=model,
                        max_tokens=4096 if detailed else 500,
                        temperature=0.5,
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    await asyncio.sleep(0.7)
                    print("Response received from Claude")
                    return response.content[0].text
                except Exception as e:
                    if "429" in str(e):
                        print("\nClaude: Hit rate limit, waiting 20 seconds...")
                        await asyncio.sleep(20)
                        response = self.claude_client.messages.create(
                            model=model,
                            max_tokens=4096 if detailed else 500,
                            temperature=0.5,
                            messages=[{
                                "role": "user",
                                "content": prompt
                            }]
                        )
                        await asyncio.sleep(0.7)
                        return response.content[0].text
                    raise e
                    
            elif self.model_choice == "gemini":
                model = "gemini-1.5-pro"  # Currently same model for both
                print(f"Using Gemini Model: {model}")
                try:
                    response = await self.gemini_model.generate_content_async(prompt)
                    await asyncio.sleep(1)  # 1 second delay for free tier
                    return response.text
                except Exception as e:
                    if "429" in str(e):
                        print("\nGemini: Hit rate limit, waiting 20 seconds...")
                        await asyncio.sleep(20)
                        response = await self.gemini_model.generate_content_async(prompt)
                        await asyncio.sleep(1)
                        return response.text
                    raise e
                    
            elif self.model_choice == "together":
                model = ("mistralai/Mixtral-8x7B-Instruct-v0.1" if detailed 
                        else "mistralai/Mistral-7B-Instruct-v0.2")
                print(f"Using Together Model: {model}")
                formatted_prompt = f"""<s>[INST] {prompt} [/INST]</s>"""
                try:
                    response = together.Complete.create(
                        prompt=formatted_prompt,
                        model=model,
                        max_tokens=4096 if detailed else 500,
                        temperature=0.5,
                        top_p=0.7,
                        top_k=50,
                        repetition_penalty=1.1
                    )
                    
                    await asyncio.sleep(0.5)
                    
                    if not response or 'choices' not in response:
                        print("No valid response or choices from Together API")
                        return None
                        
                    if response['choices'] and 'text' in response['choices'][0]:
                        return response['choices'][0]['text']
                    else:
                        print("No text in API response")
                        return None
                        
                except Exception as e:
                    if "429" in str(e):
                        print("\nTogether: Hit rate limit, waiting 20 seconds...")
                        await asyncio.sleep(20)
                        response = together.Complete.create(
                            prompt=formatted_prompt,
                            model=model,
                            max_tokens=4096 if detailed else 500,
                            temperature=0.5,
                            top_p=0.7,
                            top_k=50,
                            repetition_penalty=1.1
                        )
                        await asyncio.sleep(0.5)
                        if response['choices'] and 'text' in response['choices'][0]:
                            return response['choices'][0]['text']
                    return None
                    
            print(f"Request sent successfully")
            print(f"Prompt preview: {prompt[:200]}...")
            print('='*50)
            
        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def _extract_impact_score(self, result: str) -> int:
        """Extract impact score from AI model response"""
        try:
            if result and "**Impact**:" in result:
                score_text = result.split("**Impact**:")[1].strip()
                return int(score_text.split()[0])
            return 0
        except Exception:
            return 0

    def _parse_detailed_analysis(self, result: str) -> Dict:
        """Parse the detailed analysis response into structured data"""
        try:
            # Initialize default structure
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
                "exit_conditions": None
            }
            
            if not result:
                return parsed
                
            # Extract scores using regex for more reliable parsing
            import re
            
            # Score patterns
            score_patterns = {
                'sentiment': r'Sentiment:\s*(\d+)',
                'relevancy': r'Relevancy:\s*(\d+)',
                'reliability': r'Source Reliability:\s*(\d+)',
                'risk': r'Risk Level:\s*(\d+)',
                'impact': r'Market Impact:\s*(\d+)'
            }
            
            # Extract scores
            for key, pattern in score_patterns.items():
                if match := re.search(pattern, result):
                    parsed['scores'][key] = int(match.group(1))

            # Extract trading plan components
            if 'Trading action plan:' in result:
                # Get everything after "Trading action plan:"
                plan_text = result.split('Trading action plan:')[1].split('\n\n')[0]
                parsed['action_plan'] = plan_text.strip()
                
                # Extract position sizing
                if 'Position sizing' in result:
                    position_match = re.search(r'Position sizing.*?:(.*?)(?:\n|$)', result)
                    if position_match:
                        parsed['position_size'] = position_match.group(1).strip()
                
                # Extract review period
                if 'Review period' in result:
                    review_match = re.search(r'Review period.*?:(.*?)(?:\n|$)', result)
                    if review_match:
                        parsed['review_period'] = review_match.group(1).strip()
                
                # Extract exit conditions
                if 'Exit conditions' in result:
                    exit_section = result.split('Exit conditions:')[1].split('\n\n')[0]
                    parsed['exit_conditions'] = [
                        condition.strip('- ').strip() 
                        for condition in exit_section.split('\n') 
                        if condition.strip('- ').strip()
                    ]

            # Add raw response for reference
            parsed['raw_analysis'] = result
            
            print("\nParsed Analysis:")
            print(json.dumps(parsed, indent=2))
            
            return parsed
            
        except Exception as e:
            print(f"Parsing error: {str(e)}")
            return {
                "error": f"Failed to parse: {str(e)}",
                "raw_analysis": result
            }

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
            result = await self._call_ai_model(prompt)
            impact_score = self._extract_impact_score(result)
            print(result)
            return impact_score, result
        except Exception as e:
            print(f"Error in initial analysis: {e}")
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
            print("Sending request to LLM...")
            result = await self._call_ai_model(prompt, detailed=True)
            print(f"Raw LLM Response:\n{result}")
            
            parsed_result = self._parse_detailed_analysis(result)
            print(f"Parsed Result:\n{json.dumps(parsed_result, indent=2)}")
            
            return parsed_result
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            return {"error": str(e)}
        try:
            result = await self._call_ai_model(prompt, detailed=True)
            return self._parse_detailed_analysis(result)
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            return {"error": str(e)}

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
            "detailed_analysis": None,
            "model_used": self.model_choice
        }
        
        # If high impact, perform detailed analysis
        if impact_score >= 6:
            print(f"\nHigh impact ({impact_score}) detected - running detailed analysis...")
            detailed_result = await self.detailed_analysis(news_item, initial_result)
            print(f"Detailed analysis result: {detailed_result}")
            analysis_result["detailed_analysis"] = detailed_result
        
        return analysis_result