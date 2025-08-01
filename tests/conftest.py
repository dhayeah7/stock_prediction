import pytest
import os
import pandas as pd
from datetime import datetime
import numpy as np

@pytest.fixture(scope="session")
def test_data_dir(tmpdir_factory):
    """Create a temporary directory for test data"""
    return tmpdir_factory.mktemp("test_data")

@pytest.fixture(scope="session")
def sample_news_data():
    """Create sample news data"""
    return pd.DataFrame({
        'date': [datetime.now().strftime('%Y-%m-%d')] * 3,
        'content': [
            "Microsoft announces new product",
            "Stock market rally continues",
            "Tech sector shows growth"
        ],
        'source': ['NewsAPI', 'Economic Times', 'StockTwits']
    })

@pytest.fixture(scope="session")
def sample_stock_data():
    """Create sample stock price data"""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'Date': dates,
        'Close': np.random.uniform(100, 200, 30),
        'Volume': np.random.uniform(1000000, 5000000, 30)
    })
