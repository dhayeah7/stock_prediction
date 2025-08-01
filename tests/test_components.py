import unittest
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from config import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDataCollectors(unittest.TestCase):
    def test_newsapi(self):
        from Data_Collection import newsapi
        result = newsapi.collect_news()
        self.assertTrue(os.path.exists(result))
        df = pd.read_csv(result)
        self.assertGreater(len(df), 0)
    
    def test_economic_times(self):
        from Data_Collection import et_collector
        headlines, dates = et_collector.scrape_news()
        self.assertTrue(len(headlines) > 0)
        self.assertEqual(len(headlines), len(dates))
    
    # Add more collector tests...

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        from Data_Processing.data_processor import DataProcessor
        self.processor = DataProcessor()
    
    def test_date_standardization(self):
        test_dates = [
            '2024-03-01',
            '01-03-24',
            '2024/03/01',
            '2024-03-01 12:34:56'
        ]
        for date in test_dates:
            result = self.processor.standardize_date(date)
            self.assertEqual(result, '2024-03-01')
    
    # Add more processing tests...

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        from Data_Processing.bert_step4 import process_sentiment
        self.analyzer = process_sentiment
    
    def test_sentiment_scores(self):
        test_texts = [
            "Great news for investors!",
            "Stock price plummets",
            "Market remains stable"
        ]
        # Add sentiment analysis tests...

if __name__ == '__main__':
    unittest.main(verbosity=2)
