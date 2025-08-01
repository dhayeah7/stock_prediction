import requests
import csv
import time
import logging
from datetime import datetime, timedelta
import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_news(retries=3, delay=1):
    """Collect news from NewsAPI with retry mechanism"""
    filename = get_output_filename('newsapi')
    
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([COLUMNS['DATE'], COLUMNS['CONTENT'], COLUMNS['SOURCE']])
        
        start_date = datetime.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        
        for i in range(DEFAULT_LOOKBACK_DAYS):
            date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            
            for attempt in range(retries):
                try:
                    params = {
                        "q": f"{COMPANY_NAME} stock",
                        "from": date,
                        "to": date,
                        "sortBy": "publishedAt",
                        "apiKey": NEWSAPI_KEY,
                        "language": "en",
                        "pageSize": 100
                    }
                    
                    response = requests.get("https://newsapi.org/v2/everything", 
                                         params=params, 
                                         timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "articles" in data:
                            for article in data["articles"]:
                                content = f"{article['title']} - {article['description'] or ''}"
                                writer.writerow([date, content, "NewsAPI"])
                        break
                    elif response.status_code == 429:  # Rate limit
                        if attempt < retries - 1:
                            time.sleep(delay * (attempt + 1))
                            continue
                    else:
                        logger.error(f"API request failed with status {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Error collecting news for {date}: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay * (attempt + 1))
                    continue
                    
            time.sleep(0.1)  # Small delay between dates
    
    logger.info(f"News data saved to {filename}")
    return filename

def main():
    try:
        collect_news()
    except Exception as e:
        logger.error(f"NewsAPI collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
