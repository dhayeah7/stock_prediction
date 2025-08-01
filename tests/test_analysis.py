import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Analysis.weighted_prediction import WeightedPredictor

class TestWeightedPredictor:
    @pytest.fixture
    def predictor(self):
        return WeightedPredictor()

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        return pd.DataFrame({
            'date': dates,
            'sentiment_score': np.random.uniform(-1, 1, 30),
            'source': np.random.choice(['NewsAPI', 'StockTwits', 'Economic Times'], 30),
            'close_price': np.random.uniform(100, 200, 30)
        })

    def test_data_preparation(self, predictor, sample_data):
        X, y = predictor.prepare_data(sample_data)
        assert X is not None
        assert y is not None
        assert len(X) == len(y)

    def test_model_training(self, predictor, sample_data):
        X, y = predictor.prepare_data(sample_data)
        model = predictor.train_model(X, y)
        assert model is not None

    def test_prediction(self, predictor, sample_data):
        predictions = predictor.predict(sample_data)
        assert len(predictions) > 0
        assert all(isinstance(pred, (float, np.float32)) for pred in predictions)

    def test_performance_metrics(self, predictor, sample_data):
        metrics = predictor.calculate_metrics(sample_data)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics