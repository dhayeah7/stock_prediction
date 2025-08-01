import time
import csv
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import random
import logging
import json
import os
import sys
import traceback
from urllib3.exceptions import InsecureRequestWarning
from .utils import setup_chrome_driver, get_user_agent
from config import OUTPUT_DIR, STOCK_SYMBOL

# Suppress only the specific warning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_random_user_agent():
    """Return a random user agent string to avoid detection."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/113.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Edge/108.0.1462.76'
    ]
    return random.choice(user_agents)

def fetch_api_messages(symbol=STOCK_SYMBOL, max_count=5000, since_id=None, max_id=None):
    """
    Fetch messages using the StockTwits API with pagination support.
    Exhaustively collects messages until no more are available or max_count is reached.
    """
    logger.info(f"Attempting to fetch up to {max_count} messages for {symbol} using API...")
    
    all_messages = []
    page = 1
    max_pages = 500  # Set a higher maximum to get more historical data
    
    # Base API URL
    base_url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'application/json',
        'Connection': 'keep-alive'
    }
    
    while len(all_messages) < max_count and page <= max_pages:
        # Build request URL with pagination parameters
        params = {}
        if max_id:
            params['max'] = max_id
        elif since_id:
            params['since'] = since_id
        
        try:
            logger.info(f"Fetching page {page} from API (current count: {len(all_messages)})")
            
            # Add a random delay to avoid rate limiting
            time.sleep(random.uniform(2.0, 3.5))
            
            # Make the API request
            response = requests.get(base_url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'messages' in data and data['messages']:
                    messages = data['messages']
                    logger.info(f"Retrieved {len(messages)} messages on page {page}")
                    
                    # Filter out any messages not related to our stock symbol
                    filtered_messages = [msg for msg in messages if 
                                       msg.get('symbol', {}).get('symbol', symbol) == symbol]
                    
                    if len(filtered_messages) < len(messages):
                        logger.info(f"Filtered out {len(messages) - len(filtered_messages)} messages from other symbols")
                    
                    all_messages.extend(filtered_messages)
                    
                    # Update the max_id for the next request (pagination)
                    # Use the oldest message ID minus 1
                    if filtered_messages:
                        last_message = filtered_messages[-1]
                        max_id = last_message['id'] - 1
                    else:
                        # If we filtered out all messages, use the last raw message
                        last_message = messages[-1]
                        max_id = last_message['id'] - 1
                    
                    # If no more messages were returned or we've reached the limit, break
                    if len(messages) < 5:  # If we get fewer messages than expected, we're likely at the end
                        logger.info("Received fewer messages than expected, likely reached the end")
                        break
                    
                else:
                    logger.warning("API response did not contain messages")
                    break
                
            else:
                logger.warning(f"API request failed with status code: {response.status_code}")
                # Try with a different delay before giving up
                time.sleep(random.uniform(10.0, 15.0))
                if page > 5:  # If we've made several attempts, break out
                    break
            
            page += 1
            
            # Break if we've reached the desired count
            if len(all_messages) >= max_count:
                logger.info(f"Reached target message count: {len(all_messages)}")
                break
                
        except Exception as e:
            logger.error(f"API request failed on page {page}: {e}")
            # Try with a different delay before giving up
            time.sleep(random.uniform(10.0, 15.0))
            if page > 5:  # If we've made several attempts, break out
                break
    
    logger.info(f"Total messages fetched: {len(all_messages)}")
    return all_messages

def get_historical_data(symbol=STOCK_SYMBOL, days_back=180):
    """
    Fetch historical data for the symbol from various StockTwits endpoints.
    Only collects data for the specified symbol.
    """
    logger.info(f"Attempting to fetch historical data for {symbol} from {days_back} days back")
    
    all_messages = []
    endpoints = [
        f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}/top.json",  # Top messages
        f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}/trending.json",  # Trending messages
        f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}/recent.json"  # Recent messages
    ]
    
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'application/json'
    }
    
    for endpoint in endpoints:
        try:
            logger.info(f"Fetching from endpoint: {endpoint}")
            response = requests.get(endpoint, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'messages' in data:
                    messages = data['messages']
                    # Filter to only include messages for our symbol
                    filtered_messages = [msg for msg in messages if 
                                       msg.get('symbol', {}).get('symbol', symbol) == symbol]
                    
                    logger.info(f"Retrieved {len(filtered_messages)} messages from {endpoint}")
                    all_messages.extend(filtered_messages)
                else:
                    logger.warning(f"No messages found in {endpoint}")
            else:
                logger.warning(f"Failed to fetch from {endpoint}: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error fetching from {endpoint}: {e}")
        
        # Add a delay between requests
        time.sleep(random.uniform(2.0, 3.5))
    
    # Deduplicate messages by ID
    unique_ids = set()
    unique_messages = []
    
    for message in all_messages:
        if message['id'] not in unique_ids:
            unique_ids.add(message['id'])
            unique_messages.append(message)
    
    logger.info(f"Retrieved {len(unique_messages)} unique historical messages")
    return unique_messages

def process_api_messages(messages, symbol):
    """Process raw API messages into structured news data."""
    news_data = []
    
    for message in messages:
        try:
            # Extract data from the API response
            created_at = message.get('created_at', 'Unknown')
            body = message.get('body', '')
            username = message.get('user', {}).get('username', 'Unknown')
            
            # Skip empty messages
            if not body.strip():
                continue
            
            # Format the timestamp
            try:
                dt_obj = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
                formatted_date = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_date = created_at
            
            # Extract sentiment if available
            sentiment = 'Unknown'
            if 'entities' in message and message['entities'] is not None:
                entities = message['entities']
                if 'sentiment' in entities and entities['sentiment'] is not None:
                    sentiment = entities['sentiment'].get('basic', 'Unknown')
            
            news_item = {
                'date': formatted_date,
                'content': body,
                'source': f'StockTwits ({symbol})',
                'sentiment': sentiment
            }
            
            news_data.append(news_item)
        except Exception as e:
            logger.warning(f"Error processing message: {e}")
            continue
    
    return news_data

def save_to_csv(data, filename=None):
    """Save the scraped data to a CSV file."""
    if not data:
        logger.warning("No data to save")
        return False
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f"stocktwits_{STOCK_SYMBOL}_{len(data)}_{timestamp}.csv")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_columns = ['date', 'content', 'source']
        for col in required_columns:
            if col not in df.columns:
                if col == 'source':
                    df[col] = f'StockTwits ({STOCK_SYMBOL})'
                else:
                    df[col] = ''
        
        # Keep only the required columns for the pipeline
        df = df[required_columns]
        
        # Remove duplicates based on content
        df = df.drop_duplicates(subset=['content'])
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the scraper focused only on the configured stock symbol."""
    logger.info(f"Starting StockTwits scraper for {STOCK_SYMBOL}...")
    symbol = STOCK_SYMBOL
    target_count = 5000  # Increased target count for more data
    
    all_news_data = []
    
    try:
        # 1. First approach: Use the API with pagination for maximum data collection
        api_messages = fetch_api_messages(symbol, max_count=target_count)
        if api_messages:
            api_news_data = process_api_messages(api_messages, symbol)
            all_news_data.extend(api_news_data)
            logger.info(f"Collected {len(api_news_data)} messages via API")
        
        # 2. Second approach: Get historical data from various endpoints for the same symbol
        if len(all_news_data) < target_count:
            remaining = target_count - len(all_news_data)
            logger.info(f"Trying to collect {remaining} more messages from historical data")
            
            historical_messages = get_historical_data(symbol)
            if historical_messages:
                historical_data = process_api_messages(historical_messages, symbol)
                
                # Add only new messages to avoid duplicates
                existing_contents = set(item['content'] for item in all_news_data)
                new_historical_data = [item for item in historical_data if item['content'] not in existing_contents]
                
                all_news_data.extend(new_historical_data)
                logger.info(f"Added {len(new_historical_data)} unique messages from historical data")
    
        # Save the final data
        if all_news_data:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(OUTPUT_DIR, f"stocktwits_{symbol}_{len(all_news_data)}_{timestamp}.csv")
            
            # Ensure the output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            if save_to_csv(all_news_data, filename):
                logger.info(f"Successfully saved {len(all_news_data)} messages for {symbol}")
                return True
            else:
                logger.error("Failed to save data")
                return False
        else:
            logger.warning("No data collected")
            return False
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()