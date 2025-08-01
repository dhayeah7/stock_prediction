import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightedSentimentPredictor:
    def __init__(self, lookback_period=10):
        self.lookback_period = lookback_period
        self.model = None
        
    def prepare_data(self, stock_data, sentiment_data, test_size=0.2):
        """
        Prepare data for LSTM model with weighted sentiment scores
        """
        try:
            # Ensure date columns are datetime
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            stock_data.index = pd.to_datetime(stock_data.index)
            
            # Merge stock and sentiment data
            merged_data = pd.merge(
                stock_data.reset_index(),
                sentiment_data.groupby('date')['weighted_sentiment'].mean().reset_index(),
                left_on='Date',
                right_on='date',
                how='left'
            )
            
            # Handle missing data
            merged_data['weighted_sentiment'] = merged_data['weighted_sentiment'].fillna(method='ffill').fillna(0)
            
            # Create features with proper scaling
            scaler = MinMaxScaler()
            features = ['Close', 'Volume', 'weighted_sentiment']
            merged_data[features] = scaler.fit_transform(merged_data[features])
            
            # Create features
            X = []
            y = []
            
            for i in range(self.lookback_period, len(merged_data)):
                X.append(merged_data[features].values[i-self.lookback_period:i])
                # Predict direction (1 for up, 0 for down)
                y.append(1 if merged_data['Close'].values[i] > merged_data['Close'].values[i-1] else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error in prepare_data: {e}")
            return None, None, None, None
    
    def build_model(self, input_shape):
        """
        Build LSTM model for prediction
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """
        Train the model
        """
        self.model = self.build_model(X_train.shape[1:])
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics, y_pred

def main():
    try:
        # Load weighted sentiment data
        weighted_sentiment = pd.read_csv('Output/weighted_sentiment_scores.csv')
        weighted_sentiment['date'] = pd.to_datetime(weighted_sentiment['date'])
        
        # Load stock data
        msft = yf.Ticker("MSFT")
        stock_data = msft.history(period="1y")
        
        # Initialize predictor
        predictor = WeightedSentimentPredictor(lookback_period=10)
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(stock_data, weighted_sentiment)
        
        # Train model
        history = predictor.train(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        metrics, predictions = predictor.evaluate(X_test, y_test)
        
        # Log results
        logger.info("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric.capitalize()}: {value:.3f}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Training history
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy with Weighted Sentiment')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Predictions vs Actual
        plt.subplot(2, 1, 2)
        plt.plot(y_test, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.title('Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Direction')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('Output/weighted_prediction_results.png')
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv('Output/model_performance_metrics.csv', index=False)
        
        logger.info("\nResults saved to Output/weighted_prediction_results.png")
        logger.info("Metrics saved to Output/model_performance_metrics.csv")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
