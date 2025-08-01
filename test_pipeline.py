import os
import sys
import logging
import pandas as pd
from datetime import datetime
import unittest
import glob
from config import *

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'test_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create necessary directories
        for directory in [OUTPUT_DIR, LOG_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def test_data_collection(self):
        """Test data collection from all sources"""
        collectors = [
            ('newsapi.py', 'NewsAPI'),
            ('et_collector.py', 'Economic Times'),
            ('stocktwits_collector.py', 'StockTwits'),
            ('yahoo_finance_collector.py', 'Yahoo Finance'),
            ('reuters_collector.py', 'Reuters'),
            ('marketwatch_collector.py', 'MarketWatch')
        ]
        
        for script, source in collectors:
            with self.subTest(source=source):
                result = os.system(f'python Data_Collection/{script}')
                self.assertEqual(result, 0, f"{source} collector failed")
    
    def test_data_processing(self):
        """Test data processing"""
        result = os.system('python Data_Processing/data_processor.py')
        self.assertEqual(result, 0, "Data processing failed")
        
        # Check if output file exists and has correct format
        output_files = glob.glob(os.path.join(OUTPUT_DIR, 'merged_data_*.csv'))
        self.assertTrue(output_files, "No merged data file found")
        
        df = pd.read_csv(output_files[-1])
        required_columns = ['date', 'content', 'source']
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing column: {col}")
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        result = os.system('python Data_Processing/bert_step4.py')
        self.assertEqual(result, 0, "Sentiment analysis failed")
        
        # Check if output file exists and has correct format
        output_files = glob.glob(os.path.join(OUTPUT_DIR, 'sentiment_analyzed_*.csv'))
        self.assertTrue(output_files, "No sentiment analysis output file found")
        
        df = pd.read_csv(output_files[-1])
        required_columns = ['date', 'content', 'source', 'sentiment', 'sentiment_score']
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing column: {col}")

def main():
    unittest.main(verbosity=2)

if __name__ == "__main__":
    main()
