import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import os
import sys
from Data_Collection.newsapi_collector import NewsAPICollector
from Data_Collection.economic_times_collector import EconomicTimesCollector
from Data_Collection.stocktwits_collector import StockTwitsCollector
from config import OUTPUT_DIR
import unittest
from Data_Collection import (
    newsapi,
    et_collector,
    stocktwits_collector,
    yahoo_finance_collector,
    reuters_collector,
    marketwatch_collector
)

class TestNewsAPICollector:
    @pytest.fixture
    def collector(self):
        return NewsAPICollector()

    @pytest.fixture
    def mock_response(self):
        return {
            "status": "ok",
            "articles": [
                {
                    "title": "Test Article 1",
                    "description": "Description 1",
                    "publishedAt": "2024-04-06T10:00:00Z"
                },
                {
                    "title": "Test Article 2",
                    "description": "Description 2",
                    "publishedAt": "2024-04-06T11:00:00Z"
                }
            ]
        }

    def test_initialization(self, collector):
        assert collector.api_key is not None
        assert collector.query == "Microsoft stock"

    @patch('requests.get')
    def test_collect_news(self, mock_get, collector, mock_response):
        # Mock the API response
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        # Run collection
        output_file = collector.collect()

        # Verify output
        assert os.path.exists(output_file)
        df = pd.read_csv(output_file)
        assert len(df) == 2
        assert all(col in df.columns for col in ['date', 'content', 'source'])

    def test_error_handling(self, collector):
        with pytest.raises(Exception):
            collector.api_key = None
            collector.collect()

class TestEconomicTimesCollector:
    @pytest.fixture
    def collector(self):
        return EconomicTimesCollector()

    @patch('selenium.webdriver.Chrome')
    def test_scraping(self, mock_chrome):
        # Mock Chrome driver
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver

        # Mock HTML content
        mock_driver.page_source = """
        <div class="clr flt topicstry story_list">
            <h2>Test Headline</h2>
            <time>06 Apr, 2024, 10:00 AM IST</time>
        </div>
        """

        collector = EconomicTimesCollector()
        output_file = collector.collect()

        assert os.path.exists(output_file)
        df = pd.read_csv(output_file)
        assert 'date' in df.columns
        assert 'content' in df.columns

class TestStockTwitsCollector:
    @pytest.fixture
    def collector(self):
        return StockTwitsCollector()

    @pytest.fixture
    def mock_response(self):
        return {
            "messages": [
                {
                    "id": 1,
                    "body": "Test message 1",
                    "created_at": "2024-04-06T10:00:00Z",
                    "user": {"username": "test_user"},
                    "entities": {"sentiment": {"basic": "Bullish"}}
                }
            ]
        }

    @patch('requests.get')
    def test_collect_messages(self, mock_get, collector, mock_response):
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        output_file = collector.collect()
        
        assert os.path.exists(output_file)
        df = pd.read_csv(output_file)
        assert all(col in df.columns for col in ['date', 'content', 'sentiment'])

class TestCollectors(unittest.TestCase):
    def test_output_format(self):
        """Test if collectors produce correctly formatted output"""
        collectors = [
            (newsapi.collect_news, "NewsAPI"),
            (et_collector.scrape_news, "Economic Times"),
            (stocktwits_collector.fetch_api_messages, "StockTwits"),
            (yahoo_finance_collector.get_yahoo_news, "Yahoo Finance"),
            (reuters_collector.collect_reuters_news, "Reuters"),
            (marketwatch_collector.collect_marketwatch_news, "MarketWatch")
        ]
        
        for collector_func, source_name in collectors:
            with self.subTest(source=source_name):
                try:
                    result = collector_func()
                    if isinstance(result, str):  # If result is a filename
                        df = pd.read_csv(result)
                    else:  # If result is data
                        df = pd.DataFrame(result)
                    
                    # Check required columns
                    self.assertIn('date', df.columns)
                    self.assertIn('content', df.columns)
                    self.assertIn('source', df.columns)
                    
                    # Check data types
                    self.assertTrue(all(isinstance(x, str) for x in df['date']))
                    self.assertTrue(all(isinstance(x, str) for x in df['content']))
                    
                except Exception as e:
                    self.fail(f"{source_name} collector failed: {e}")

if __name__ == '__main__':
    unittest.main()
