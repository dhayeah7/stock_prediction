# run_pipeline.py - modified

import os
import logging
from datetime import datetime
import sys
import glob
import signal
import time
from Data_Collection import (
    newsapi, 
    stocktwits_collector,
    yahoo_finance_collector,
    google_collector
)
from Data_Processing import data_processor, bert_step4
from config import OUTPUT_DIR, LOG_DIR
from Data_Processing.data_validator import DataValidator

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'pipeline_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add a timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=60):
    """Run a function with a timeout"""
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Disable the alarm
        return result
    except TimeoutError:
        logger.error(f"Function {func.__name__} timed out after {timeout_duration} seconds")
        return None
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'pandas': 'Data processing',
        'requests': 'API requests',
        'feedparser': 'RSS feed parsing'
    }
    
    missing_packages = []
    for package, purpose in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(f"{package} (needed for {purpose})")
    
    if missing_packages:
        logger.error("Missing required packages:")
        for package in missing_packages:
            logger.error(f"  - {package}")
        logger.error("\nPlease install missing packages using:")
        logger.error("pip install -r requirements.txt")
        sys.exit(1)

def run_pipeline():
    logger.info("Starting pipeline...")
    try:
        check_dependencies()
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        return
    
    # Initialize validator
    try:
        validator = DataValidator()
    except Exception as e:
        logger.error(f"Error initializing validator: {e}")
        return
    
    # Use only simpler collectors that don't rely on Selenium
    collectors = [
        #(newsapi.main, "NewsAPI"),
        #(stocktwits_collector.main, "StockTwits"),
        #(yahoo_finance_collector.main, "Yahoo Finance"),
        #(google_collector.main, "Google News")
    ]
    
    for collector_func, source_name in collectors:
        try:
            logger.info(f"Collecting data from {source_name}...")
            
            # Run the collector with a timeout
            result = run_with_timeout(collector_func, timeout_duration=300)  # 5 minutes timeout
            
            if result is None:
                logger.error(f"Collection from {source_name} timed out")
                continue
            
            # Validate collected data
            pattern = os.path.join(OUTPUT_DIR, f"*{source_name.lower().replace(' ', '_')}*.csv")
            matching_files = glob.glob(pattern)
            if not matching_files:
                logger.error(f"No output files found for {source_name}")
                continue
                
            latest_file = max(matching_files, key=os.path.getctime)
            
            # Run validation with a timeout
            validation_result = run_with_timeout(
                validator.validate_file, 
                args=(latest_file,), 
                timeout_duration=60
            )
            
            if validation_result:
                logger.info(f"Successfully collected and validated data from {source_name}")
            else:
                logger.error(f"Validation failed for {source_name} data")
                continue
                
        except Exception as e:
            logger.error(f"Error collecting from {source_name}: {e}")
    
    logger.info("Processing and combining data...")
    try:
        processor = data_processor.DataProcessor()
        output_file = run_with_timeout(
            processor.process_all_data, 
            timeout_duration=120
        )
        
        if output_file:
            logger.info(f"Data processing completed. Output saved to: {output_file}")
        else:
            logger.error("Data processing timed out or failed")
            return
            
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        return
    
    print("\n=== Running Sentiment Analysis ===")
    try:
        print("Starting sentiment analysis...")
        sentiment_result = run_with_timeout(
            bert_step4.process_sentiment,
            kwargs={"input_file": output_file},
            timeout_duration=1800  # Increased to 30 minutes
        )
        
        if sentiment_result is not None:
            print("Sentiment analysis completed successfully")
        else:
            print("Sentiment analysis timed out")
            return
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
