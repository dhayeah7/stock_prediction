import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import time
import random
from bs4 import BeautifulSoup
from config import OUTPUT_DIR, STOCK_SYMBOL, COMPANY_NAME

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
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/108.0.1462.76'
    ]
    return random.choice(user_agents)

def fetch_google_news(query, max_results=200, time_period=None):
    """
    Fetch news from Google News RSS feed with time period filtering.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to fetch per query
        time_period: Optional time period string ('1h', '1d', '7d', '1m', '1y')
    """
    news_items = []
    
    # Add time period to query if specified
    encoded_query = query.replace(' ', '+')
    if time_period:
        encoded_query += f"+when:{time_period}"
    
    # Try different Google News URLs to maximize results
    urls = [
        f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/news/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/headlines/section/topic/BUSINESS?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    ]
    
    for url in urls:
        try:
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'application/xml,application/xhtml+xml,text/html',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://news.google.com/'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                logger.info(f"Found {len(items)} news items for query: {query} at {url}")
                
                for item in items[:max_results]:
                    title = item.title.text
                    pub_date = item.pubDate.text
                    
                    try:
                        date_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                        formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    news_items.append({
                        'date': formatted_date,
                        'content': title,
                        'source': 'Google News'  # Standardized source name
                    })
            else:
                logger.warning(f"Failed to fetch news from {url}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching news from {url}: {e}")
        
        # Add delay between requests
        time.sleep(random.uniform(2, 4))
    
    return news_items

def save_to_csv(data, filename=None):
    """Save the collected news data to a CSV file."""
    if not data:
        logger.warning("No data to save")
        return False
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f"google_news_{timestamp}.csv")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Remove duplicates based on content
        df = df.drop_duplicates(subset=['content'])
        
        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        return False

def main():
    """Main function to collect news from Google News."""
    logger.info(f"Starting Google News collector for {COMPANY_NAME} ({STOCK_SYMBOL})...")
    
    # Expanded query list with variations to get more results
    queries = [
        STOCK_SYMBOL,
        COMPANY_NAME,
        f"{STOCK_SYMBOL} stock",
        f"{COMPANY_NAME} stock",
        f"{STOCK_SYMBOL} market",
        f"{COMPANY_NAME} market",
        f"{STOCK_SYMBOL} financial",
        f"{COMPANY_NAME} financial",
        f"{STOCK_SYMBOL} earnings",
        f"{COMPANY_NAME} earnings"
    ]
    
    # Time periods to search across
    time_periods = ['1h', '1d', '7d', '1m', '1y']
    
    all_news_items = []
    target_count = 2000
    
    # First try without time period filtering
    for query in queries:
        if len(all_news_items) >= target_count:
            break
            
        news_items = fetch_google_news(query, max_results=200)
        all_news_items.extend(news_items)
        
        # Add delay between queries
        time.sleep(random.uniform(3, 5))
    
    # If we still need more items, try with time period filtering
    if len(all_news_items) < target_count:
        for period in time_periods:
            if len(all_news_items) >= target_count:
                break
                
            for query in queries:
                if len(all_news_items) >= target_count:
                    break
                    
                news_items = fetch_google_news(query, max_results=200, time_period=period)
                all_news_items.extend(news_items)
                
                # Add delay between queries
                time.sleep(random.uniform(3, 5))
    
    if all_news_items:
        # Remove duplicates
        df = pd.DataFrame(all_news_items)
        df = df.drop_duplicates(subset=['content'])
        unique_items = df.to_dict('records')
        
        logger.info(f"Collected {len(unique_items)} unique news items")
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f"google_news_{len(unique_items)}_{timestamp}.csv")
        
        return save_to_csv(unique_items, filename)
    else:
        logger.warning("No news items collected")
        return False

if __name__ == "__main__":
    main()