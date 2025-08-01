import pandas as pd
import os
import glob
from datetime import datetime
import logging
import re
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean text by removing URLs, special characters, and extra whitespace"""
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def combine_news_sources(output_dir='Output', clean_data=True):
    """
    Combine all news sources into a single DataFrame and optionally clean the data
    
    Args:
        output_dir (str): Directory containing the news data files
        clean_data (bool): Whether to clean the text content
    """
    try:
        all_news = []
        
        # Get all CSV files from the output directory
        csv_files = glob.glob(os.path.join(output_dir, '*.csv'))
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                
                # Ensure required columns exist
                if not all(col in df.columns for col in ['date', 'content']):
                    logger.warning(f"Skipping {file} - missing required columns")
                    continue
                
                # Determine source based on filename or existing source column
                if 'source' not in df.columns:
                    if 'newsapi' in file.lower():
                        df['source'] = 'NewsAPI'
                    elif 'stocktwits' in file.lower():
                        df['source'] = 'StockTwits'
                    elif 'economic_times' in file.lower() or 'et_' in file.lower():
                        df['source'] = 'Economic Times'
                    elif 'yahoo' in file.lower():
                        df['source'] = 'Yahoo Finance'
                    elif 'google' in file.lower():
                        df['source'] = 'Google News'
                    else:
                        df['source'] = 'Other'
                
                # Clean content if requested
                if clean_data:
                    df['content'] = df['content'].astype(str).apply(clean_text)
                    # Remove rows with empty content after cleaning
                    df = df[df['content'].str.strip() != '']
                
                # Keep only required columns
                df = df[['date', 'content', 'source']]
                
                all_news.append(df)
                logger.info(f"Successfully processed {file}")
                
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                continue
        
        if not all_news:
            logger.error("No valid news data files found!")
            return None
        
        # Combine all sources
        combined_news = pd.concat(all_news, ignore_index=True)
        
        # Remove duplicates based on content
        combined_news = combined_news.drop_duplicates(subset=['content'])
        
        # Standardize date format
        combined_news['date'] = pd.to_datetime(combined_news['date'])
        
        # Sort by date
        combined_news.sort_values('date', inplace=True)
        
        # Convert date back to string format for consistency
        combined_news['date'] = combined_news['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save combined and cleaned data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'combined_cleaned_news_{timestamp}.csv')
        combined_news.to_csv(output_file, index=False)
        logger.info(f"Combined and cleaned news saved to {output_file}")
        logger.info(f"Total rows in combined file: {len(combined_news)}")
        
        return combined_news
        
    except Exception as e:
        logger.error(f"Error combining news sources: {e}")
        return None

def main():
    combine_news_sources(clean_data=True)

if __name__ == "__main__":
    main()
