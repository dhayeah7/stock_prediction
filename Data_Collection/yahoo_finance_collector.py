import csv
import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import logging
import os
from config import *
import time
import random
import re

logger = logging.getLogger(__name__)

headers = {
    'accept': '*/*',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.google.com',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
}

def parse_relative_time(time_str):
    """Convert relative time string to datetime object"""
    now = datetime.now()
    time_str = time_str.lower().strip()
    
    try:
        # Handle "X minutes/hours ago"
        if 'minute' in time_str:
            minutes = int(re.findall(r'\d+', time_str)[0])
            return now - timedelta(minutes=minutes)
        elif 'hour' in time_str:
            hours = int(re.findall(r'\d+', time_str)[0])
            return now - timedelta(hours=hours)
        # Handle "X days ago"
        elif 'day' in time_str:
            days = int(re.findall(r'\d+', time_str)[0])
            return now - timedelta(days=days)
        # Handle "X months ago"
        elif 'month' in time_str:
            months = int(re.findall(r'\d+', time_str)[0])
            return now - timedelta(days=months*30)
        # Handle "X years ago"
        elif 'year' in time_str:
            years = int(re.findall(r'\d+', time_str)[0])
            return now - timedelta(days=years*365)
        else:
            return now
    except:
        logger.warning(f"Could not parse time string: {time_str}")
        return now

def get_random_delay():
    """Return a random delay between 1 and 3 seconds"""
    return random.uniform(1, 3)

def get_yahoo_news(symbol, max_articles=1000, batch_save_size=100):
    """Scrape Yahoo News headlines for a given stock symbol."""
    template = 'https://news.search.yahoo.com/search?p={}+stock+market+news'
    url = template.format(symbol)
    news_items = []
    processed_urls = set()  # To avoid duplicates
    page_count = 0
    max_pages = 100  # Safety limit
    
    logger.info(f"Starting Yahoo Finance news collection for {symbol}, targeting {max_articles} articles")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output_file = os.path.join(OUTPUT_DIR, f"yahoo_finance_final_{timestamp}.csv")
    
    while len(news_items) < max_articles and page_count < max_pages:
        try:
            logger.info(f"Scraping page {page_count + 1}, current articles: {len(news_items)}")
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.warning(f"Got status code {response.status_code}, waiting longer before retry...")
                time.sleep(get_random_delay() * 2)
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            cards = soup.find_all('div', 'NewsArticle')
            
            if not cards:
                logger.warning("No news articles found on page")
                break
            
            new_articles_on_page = 0
            
            for card in cards:
                try:
                    # Get headline
                    headline_elem = card.find('h4', class_='s-title')
                    if not headline_elem:
                        continue
                    headline = headline_elem.get_text(strip=True)
                    
                    # Get description
                    description = card.find('p', class_='s-desc')
                    description_text = description.get_text(strip=True) if description else ""
                    
                    # Get and parse publication date
                    date_elem = card.find('span', class_='s-time')
                    if date_elem:
                        date_text = date_elem.get_text(strip=True).replace('·', '').strip()
                        pub_date = parse_relative_time(date_text)
                    else:
                        pub_date = datetime.now()
                    
                    # Get source
                    source_elem = card.find('span', class_='s-source')
                    source = source_elem.get_text(strip=True) if source_elem else "Yahoo Finance"
                    
                    # Get URL
                    url_elem = card.find('a')
                    article_url = url_elem.get('href') if url_elem else ""
                    
                    # Skip if we've seen this URL before
                    if article_url in processed_urls:
                        continue
                    processed_urls.add(article_url)
                    
                    # Combine headline and description
                    content = f"{headline}. {description_text}"
                    
                    news_items.append({
                        'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'content': content,
                        'source': 'Yahoo Finance',
                        'url': article_url
                    })
                    
                    new_articles_on_page += 1
                    
                    # Batch save every batch_save_size articles
                    if len(news_items) % batch_save_size == 0:
                        logger.info(f"Batch saving at {len(news_items)} articles...")
                        df_batch = pd.DataFrame(news_items)
                        # Sort by date before saving
                        df_batch['date'] = pd.to_datetime(df_batch['date'])
                        df_batch = df_batch.sort_values('date', ascending=False)
                        df_batch['date'] = df_batch['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        df_batch.to_csv(final_output_file, index=False)
                    
                    if len(news_items) >= max_articles:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing news card: {e}")
                    continue
            
            logger.info(f"Found {new_articles_on_page} new articles on page {page_count + 1}")
            
            if new_articles_on_page == 0:
                logger.warning("No new articles found on this page, might be at the end")
                break
            
            # Check for next page
            next_page = soup.find('a', class_='next')
            if next_page and next_page.get('href'):
                url = next_page.get('href')
                page_count += 1
                delay = get_random_delay()
                logger.info(f"Moving to next page, waiting {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.info("No more pages available")
                break
                
        except Exception as e:
            logger.error(f"Error scraping page: {e}")
            if news_items:
                df = pd.DataFrame(news_items)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False)
                df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                df.to_csv(final_output_file, index=False)
            break
    
    if not news_items:
        logger.warning("No news items collected")
        return None
    
    # Final save with sorting by date
    df = pd.DataFrame(news_items)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(final_output_file, index=False)
    logger.info(f"Saved {len(df)} Yahoo Finance news items to {final_output_file}")
    
    return final_output_file

def main():
    """Main function to run the Yahoo Finance collector"""
    try:
        output_file = get_yahoo_news(STOCK_SYMBOL, max_articles=1000)
        
        if output_file and os.path.exists(output_file):
            logger.info("Yahoo Finance collection completed successfully")
            return True
        else:
            logger.error("Yahoo Finance collection failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False

if __name__ == "__main__":
    main()