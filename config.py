import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Directory paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'Output')
DATA_COLLECTION_DIR = os.path.join(ROOT_DIR, 'Data_Collection')
DATA_PROCESSING_DIR = os.path.join(ROOT_DIR, 'Data_Processing')
ANALYSIS_DIR = os.path.join(ROOT_DIR, 'Analysis')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

# Ensure directories exist
for directory in [OUTPUT_DIR, DATA_COLLECTION_DIR, DATA_PROCESSING_DIR, ANALYSIS_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load environment variables
load_dotenv()

# API Configuration
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL', 'MSFT')
COMPANY_NAME = os.getenv('COMPANY_NAME', 'Microsoft')

# Time periods
DEFAULT_LOOKBACK_DAYS = 30
MAX_LOOKBACK_DAYS = 365

# Model parameters
BERT_MODEL_NAME = "yiyanghkust/finbert-tone"
LSTM_LOOKBACK_DAYS = 10
TRAIN_TEST_SPLIT = 0.2

# Source credibility weights
SOURCE_WEIGHTS = {
    'NewsAPI': 0.698865,
    'Economic Times': 0.655000,
    'StockTwits': 0.692267,
    'Yahoo Finance': 0.847193,
    'Google News': 0.646000
}

# File naming
def get_output_filename(prefix, extension='csv'):
    """Generate standardized output filename"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(OUTPUT_DIR, f"{prefix}_{timestamp}.{extension}")

# Common column names
COLUMNS = {
    'DATE': 'date',
    'CONTENT': 'content',
    'SOURCE': 'source',
    'SENTIMENT': 'sentiment_score',
    'WEIGHTED_SENTIMENT': 'weighted_sentiment'
}

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOG_DIR, f'pipeline_{datetime.now().strftime("%Y%m%d")}.log')

# API endpoints
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
ECONOMIC_TIMES_URL = f"https://economictimes.indiatimes.com/topic/stock-{COMPANY_NAME.lower()}"
STOCKTWITS_API = f"https://api.stocktwits.com/api/2/streams/symbol/{STOCK_SYMBOL}.json"

# Data validation
REQUIRED_COLUMNS = {
    'newsapi': ['date', 'content', 'source'],
    'economic_times': ['date', 'content', 'source'],
    'stocktwits': ['date', 'content', 'source', 'sentiment'],
    'yahoo_finance': ['date', 'content', 'source', 'url'],
    'google_news': ['date', 'content', 'source', 'url']
}

# Analysis parameters
ANALYSIS_PARAMS = {
    'lstm_units': 50,
    'dropout_rate': 0.2,
    'dense_units': 25,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.1
}
