import logging
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import random

def setup_chrome_driver(headless=True):
    """Common function to set up Chrome WebDriver"""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f'user-agent={get_user_agent()}')
    
    try:
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)
    except Exception as e:
        logging.error(f"Failed to initialize WebDriver: {e}")
        raise

def get_user_agent():
    """Return a common user agent string"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/113.0'
    ]
    return random.choice(user_agents)

def standardize_date(date_str):
    """Standardize date string to YYYY-MM-DD format"""
    try:
        for fmt in ['%Y-%m-%d', '%d-%m-%y', '%Y/%m/%d', 
                   '%d/%m/%y', '%m/%d/%y', '%y-%m-%d',
                   '%Y-%m-%d %H:%M:%S', '%B %d, %Y']:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        if 'T' in date_str:
            return date_str.split('T')[0]
            
        raise ValueError(f"Unable to parse date: {date_str}")
    except Exception as e:
        logging.warning(f"Date parsing failed for {date_str}: {e}")
        return None

def clean_text(text):
    """Clean and standardize text content"""
    if pd.isna(text):
        return ""
    return ' '.join(str(text).split())
