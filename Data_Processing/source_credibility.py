# Data Processing/source_credibility.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceCredibilityAnalyzer:
    def __init__(self):
        self.source_metrics = {
            'NewsAPI': {
                'domain_authority': 0.8,  # Initial domain authority scores
                'historical_accuracy': None,
                'audience_reach': 0.75
            },
            'StockTwits': {
                'domain_authority': 0.6,
                'historical_accuracy': None,
                'audience_reach': 0.7
            },
            'Economic Times': {
                'domain_authority': 0.85,
                'historical_accuracy': None,
                'audience_reach': 0.8
            }
        }
        self.alpha = 0.6  # Weight for historical accuracy
        self.beta = 0.4   # Weight for domain authority

    def calculate_historical_accuracy(self, news_data, stock_data, window=3):
        """
        Calculate historical accuracy for each source based on price movements
        """
        accuracies = {}
        
        for source in news_data['source'].unique():
            source_news = news_data[news_data['source'] == source].copy()
            correct_predictions = 0
            total_predictions = 0
            
            for _, article in source_news.iterrows():
                try:
                    # Get stock movement after news
                    article_date = pd.to_datetime(article['date'])
                    future_date = article_date + pd.Timedelta(days=window)
                    
                    # Get relevant stock prices
                    start_price = stock_data.loc[stock_data.index >= article_date]['Close'].iloc[0]
                    future_price = stock_data.loc[stock_data.index >= future_date]['Close'].iloc[0]
                    
                    # Calculate return
                    actual_return = (future_price - start_price) / start_price
                    
                    # Compare sentiment with actual movement
                    if (article['sentiment_score'] > 0 and actual_return > 0) or \
                       (article['sentiment_score'] < 0 and actual_return < 0):
                        correct_predictions += 1
                    
                    total_predictions += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing article from {source}: {e}")
                    continue
            
            if total_predictions > 0:
                accuracies[source] = correct_predictions / total_predictions
            else:
                accuracies[source] = 0.5  # Default accuracy if no predictions
                
        return accuracies

    def calculate_source_credibility_index(self, historical_accuracies):
        """
        Calculate the Source Credibility Index (SCI) for each source
        """
        scis = {}
        
        for source, metrics in self.source_metrics.items():
            # Update historical accuracy
            metrics['historical_accuracy'] = historical_accuracies.get(source, 0.5)
            
            # Calculate SCI using weighted formula
            sci = (self.alpha * metrics['historical_accuracy'] + 
                  self.beta * metrics['domain_authority'])
            
            scis[source] = sci
            
        # Normalize SCIs to 0-1 range
        scaler = MinMaxScaler()
        sci_values = np.array(list(scis.values())).reshape(-1, 1)
        normalized_scis = scaler.fit_transform(sci_values).flatten()
        
        return dict(zip(scis.keys(), normalized_scis))

    def apply_credibility_weights(self, news_data, credibility_scores):
        """
        Apply credibility weights to sentiment scores
        """
        news_data = news_data.copy()
        
        # Apply weights to sentiment scores
        news_data['weighted_sentiment'] = news_data.apply(
            lambda row: row['sentiment_score'] * credibility_scores.get(row['source'], 0.5),
            axis=1
        )
        
        return news_data

def main():
    # Load the collected news data
    try:
        # Load and combine news from different sources
        newsapi_data = pd.read_csv('Output/MSFT_newsapi_latest.csv')
        stocktwits_data = pd.read_csv('Output/MSFT_stocktwits_latest.csv')
        economic_times_data = pd.read_csv('Output/news_headlines.csv')
        
        # Combine all news sources
        all_news = pd.concat([
            newsapi_data,
            stocktwits_data,
            economic_times_data
        ], ignore_index=True)
        
        # Load stock data
        msft = yf.Ticker("MSFT")
        stock_data = msft.history(period="1y")
        
        # Initialize analyzer
        analyzer = SourceCredibilityAnalyzer()
        
        # Calculate historical accuracy
        historical_accuracies = analyzer.calculate_historical_accuracy(all_news, stock_data)
        logger.info("Historical accuracies calculated:")
        for source, accuracy in historical_accuracies.items():
            logger.info(f"{source}: {accuracy:.3f}")
        
        # Calculate credibility scores
        credibility_scores = analyzer.calculate_source_credibility_index(historical_accuracies)
        logger.info("\nCredibility scores calculated:")
        for source, score in credibility_scores.items():
            logger.info(f"{source}: {score:.3f}")
        
        # Apply weights to sentiment scores
        weighted_news = analyzer.apply_credibility_weights(all_news, credibility_scores)
        
        # Save results
        weighted_news.to_csv('Output/weighted_sentiment_scores.csv', index=False)
        
        # Save credibility metrics
        pd.DataFrame({
            'Source': credibility_scores.keys(),
            'Credibility_Score': credibility_scores.values(),
            'Historical_Accuracy': [historical_accuracies.get(source, 0) for source in credibility_scores.keys()]
        }).to_csv('Output/source_credibility_metrics.csv', index=False)
        
        logger.info("\nResults saved to Output/weighted_sentiment_scores.csv")
        logger.info("Credibility metrics saved to Output/source_credibility_metrics.csv")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()