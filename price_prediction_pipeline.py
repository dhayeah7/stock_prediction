import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import ta  # Technical Analysis library
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Setup logging and directories
OUTPUT_DIR = 'Output'
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f'price_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_historical_data(symbol, years=5):
    """Load extensive historical data with calendar and seasonal patterns"""
    try:
        print(f"Loading {years} years of historical data for {symbol}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # Load stock data
        stock = yf.Ticker(symbol)
        hist_data = stock.history(start=start_date, end=end_date)
        
        # Add calendar features
        hist_data['Year'] = hist_data.index.year
        hist_data['Month'] = hist_data.index.month
        hist_data['DayOfWeek'] = hist_data.index.dayofweek
        hist_data['DayOfYear'] = hist_data.index.dayofyear
        hist_data['WeekOfYear'] = hist_data.index.isocalendar().week
        
        # Add seasonal patterns
        hist_data['IsMonthStart'] = hist_data.index.is_month_start.astype(int)
        hist_data['IsMonthEnd'] = hist_data.index.is_month_end.astype(int)
        hist_data['IsQuarterEnd'] = hist_data.index.is_quarter_end.astype(int)
        hist_data['IsYearEnd'] = hist_data.index.is_year_end.astype(int)
        
        return hist_data
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        raise

class EnhancedStockPricePredictor:
    def __init__(self, lookback_days=60, prediction_days=5):
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.models = []
        self.price_scaler = RobustScaler()
        self.feature_scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def add_historical_patterns(self, df):
        """Add historical pattern indicators"""
        # Monthly patterns
        df['monthly_avg'] = df.groupby('Month')['Close'].transform('mean')
        df['monthly_std'] = df.groupby('Month')['Close'].transform('std')
        df['price_vs_monthly_avg'] = df['Close'] / df['monthly_avg']
        
        # Weekly patterns
        df['weekly_avg'] = df.groupby('WeekOfYear')['Close'].transform('mean')
        df['weekly_std'] = df.groupby('WeekOfYear')['Close'].transform('std')
        df['price_vs_weekly_avg'] = df['Close'] / df['weekly_avg']
        
        # Day of week patterns
        df['dow_avg'] = df.groupby('DayOfWeek')['Close'].transform('mean')
        df['dow_std'] = df.groupby('DayOfWeek')['Close'].transform('std')
        df['price_vs_dow_avg'] = df['Close'] / df['dow_avg']
        
        # Yearly patterns
        df['yearly_avg'] = df.groupby('Year')['Close'].transform('mean')
        df['yearly_std'] = df.groupby('Year')['Close'].transform('std')
        df['price_vs_yearly_avg'] = df['Close'] / df['yearly_avg']
        
        return df
    
    def add_advanced_technical_indicators(self, df):
        """Add advanced technical analysis indicators"""
        # Trend Indicators: Simple and Exponential Moving Averages
        for window in [10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window, min_periods=1).mean()
        
        df['macd'] = ta.trend.macd_diff(df['Close']).fillna(0)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close']).fillna(0)
        df['kst'] = ta.trend.kst(df['Close']).fillna(0)
        
        # Momentum Indicators
        df['rsi'] = ta.momentum.rsi(df['Close']).fillna(50)
        df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close']).fillna(50)
        df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close']).fillna(0)
        df['roc'] = ta.momentum.roc(df['Close']).fillna(0)
        df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close']).fillna(0)
        
        # Volatility Indicators
        df['bbands_upper'] = ta.volatility.bollinger_hband(df['Close']).fillna(method='bfill')
        df['bbands_lower'] = ta.volatility.bollinger_lband(df['Close']).fillna(method='bfill')
        df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close']).fillna(0)
        df['keltner_upper'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close']).fillna(method='bfill')
        df['keltner_lower'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close']).fillna(method='bfill')
        
        # Volume Indicators
        df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume']).fillna(method='bfill')
        df['cmf'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume']).fillna(0)
        df['fi'] = ta.volume.force_index(df['Close'], df['Volume']).fillna(0)
        df['em'] = ta.volume.ease_of_movement(df['High'], df['Low'], df['Volume']).fillna(0)
        df['vwap'] = df['Close']  # vwap remains trivial as in original
        
        return df
    
    def add_price_patterns(self, df):
        """Add candlestick pattern recognition features"""
        df['body'] = df['Close'] - df['Open']
        df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Simple candlestick pattern flags
        df['doji'] = (abs(df['body']) <= 0.1 * (df['High'] - df['Low'])).astype(int)
        df['hammer'] = ((df['lower_shadow'] > 2 * abs(df['body'])) & (df['upper_shadow'] <= abs(df['body']))).astype(int)
        df['shooting_star'] = ((df['upper_shadow'] > 2 * abs(df['body'])) & (df['lower_shadow'] <= abs(df['body']))).astype(int)
        
        return df

    def train(self, X_train, y_train, model_type='gb'):
        """Train separate models for each prediction day with proper imputation"""
        print(f"\nTraining {model_type.upper()} models for each prediction day...")
        self.models = []
        
        # First, impute any missing values in X_train
        X_train_imputed = self.imputer.fit_transform(X_train)
        
        for day in range(self.prediction_days):
            if model_type == 'gb':
                model = Pipeline([
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=4,
                        min_samples_split=5,
                        min_samples_leaf=3,
                        subsample=0.8,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=42
                    ))
                ])
            elif model_type == 'rf':
                model = Pipeline([
                    ('regressor', RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    ))
                ])
            
            # Train model for specific day
            y_train_day = y_train[:, day]
            model.fit(X_train_imputed, y_train_day)
            self.models.append(model)
            print(f"Model for day {day + 1} trained successfully")
    
    def predict(self, X):
        """Make predictions for all days with proper imputation"""
        # Impute any missing values before prediction
        X_imputed = self.imputer.transform(X)
        predictions = np.zeros((X.shape[0], self.prediction_days))
        for day in range(self.prediction_days):
            predictions[:, day] = self.models[day].predict(X_imputed)
        return predictions
    
    def evaluate(self, X, y_true):
        """Evaluate model performance for each prediction day"""
        y_pred = self.predict(X)
        metrics = []
        
        for day in range(self.prediction_days):
            day_metrics = {
                'day': day + 1,
                'rmse': np.sqrt(mean_squared_error(y_true[:, day], y_pred[:, day])),
                'r2': r2_score(y_true[:, day], y_pred[:, day]),
                'mape': mean_absolute_percentage_error(y_true[:, day], y_pred[:, day])
            }
            metrics.append(day_metrics)
        
        return metrics

    def prepare_data(self, stock_data, sentiment_data, test_split=0.2, use_weights=True):
        """Prepare data with robust NaN handling"""
        try:
            # Convert stock data index to timezone-naive datetime
            if stock_data.index.tzinfo is not None:
                stock_data.index = stock_data.index.tz_localize(None)
            
            # Add technical and historical indicators
            stock_data = self.add_historical_patterns(stock_data)
            stock_data = self.add_advanced_technical_indicators(stock_data)
            stock_data = self.add_price_patterns(stock_data)
            
            # Ensure no NaNs in stock data
            stock_data = stock_data.fillna(0)
            
            # Ensure sentiment_data 'date' column is datetime
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            
            # Debug: Print column names to verify
            print("Sentiment DataFrame columns:", sentiment_data.columns)
            
            if not use_weights:
                # When weights are not used, switch to numeric sentiment
                sentiment_data['weighted_sentiment'] = sentiment_data['sentiment_numeric']
            
            # Merge sentiment data with stock data
            merged_data = pd.merge(
                stock_data.reset_index(),
                sentiment_data,
                left_on='Date',
                right_on='date',
                how='left'
            )
            
            # Drop the redundant date column and set index
            if 'date' in merged_data.columns:
                merged_data = merged_data.drop('date', axis=1)
            
            merged_data.set_index('Date', inplace=True)
            
            # Fill sentiment nulls with 0
            merged_data['weighted_sentiment'] = merged_data['weighted_sentiment'].fillna(0)
            
            # Add sentiment features over different windows
            for window in [5, 10, 20]:
                merged_data[f'sentiment_ma{window}'] = merged_data['weighted_sentiment'].rolling(window=window, min_periods=1).mean().fillna(0)
            
            # Final check for any remaining NaNs
            merged_data = merged_data.fillna(0)
            
            # Exclude date-related columns from features
            exclude_cols = ['Year', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']
            feature_columns = [col for col in merged_data.columns if col not in exclude_cols]
            
            # Prepare input features X and target y
            X, y = [], []
            
            # Use a sliding window (lookback) approach to extract features
            for i in range(self.lookback_days, len(merged_data) - self.prediction_days + 1):
                features = merged_data[feature_columns].iloc[i - self.lookback_days:i].values.flatten()
                X.append(features)
                # Get the next prediction_days of closing prices as target
                y.append(merged_data['Close'].iloc[i:i + self.prediction_days].values)
            
            X = np.array(X)
            y = np.array(y)
            
            # Print shapes for debugging
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            
            # Scale features and target
            X = self.feature_scaler.fit_transform(X)
            y = self.price_scaler.fit_transform(y)
            
            # Split data into training and test sets
            train_size = int(len(X) * (1 - test_split))
            X_train = X[:train_size]
            X_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            return X_train, X_test, y_train, y_test, merged_data
            
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise

def run_price_prediction():
    try:
        print("\n=== Starting Enhanced Price Prediction Pipeline ===\n")
        
        # Step 1: Load sentiment data
        print("Loading sentiment data...")
        # Use the provided file path
        sentiment_file = 'output/sentiment_analyzed_news.csv'
        sentiment_df = pd.read_csv(sentiment_file)
        
        # Debug: Print column names and first few rows
        print("Sentiment DataFrame columns:", sentiment_df.columns)
        print("First few rows of sentiment data:")
        print(sentiment_df.head())
        
        # Convert sentiment to numeric if applicable
        sentiment_mapping = {'positive': 0.5, 'neutral': 0, 'negative': -0.5}
        sentiment_df['sentiment_numeric'] = sentiment_df['sentiment'].map(sentiment_mapping)
        
        # Aggregate sentiment by date
        daily_sentiment = sentiment_df.groupby('date').agg({
            'weighted_sentiment': 'mean',
            'sentiment_numeric': 'mean'
        }).reset_index()
        
        # Step 2: Load historical stock data
        print("\nLoading historical stock data...")
        hist_data = load_historical_data("MSFT", years=5)
        
        # Step 3: Initialize and train models
        print("\nInitializing price predictor...")
        predictor = EnhancedStockPricePredictor(lookback_days=30, prediction_days=5)
        
        print("\nPreparing data with weights...")
        X_train_w, X_test_w, y_train_w, y_test_w, merged_data_w = predictor.prepare_data(
            hist_data,
            daily_sentiment,
            use_weights=True
        )
        
        print("\nTraining Gradient Boosting model with weights...")
        predictor.train(X_train_w, y_train_w, model_type='gb')
        
        print("\nMaking predictions with weights...")
        train_predictions_w = predictor.predict(X_train_w)
        test_predictions_w = predictor.predict(X_test_w)
        
        print("\nEvaluating model with weights...")
        train_metrics_w = predictor.evaluate(X_train_w, y_train_w)
        test_metrics_w = predictor.evaluate(X_test_w, y_test_w)
        
        print("\nPreparing data without weights...")
        X_train_uw, X_test_uw, y_train_uw, y_test_uw, merged_data_uw = predictor.prepare_data(
            hist_data,
            daily_sentiment,
            use_weights=False
        )
        
        print("\nTraining Gradient Boosting model without weights...")
        predictor.train(X_train_uw, y_train_uw, model_type='gb')
        
        print("\nMaking predictions without weights...")
        train_predictions_uw = predictor.predict(X_train_uw)
        test_predictions_uw = predictor.predict(X_test_uw)
        
        print("\nEvaluating model without weights...")
        train_metrics_uw = predictor.evaluate(X_train_uw, y_train_uw)
        test_metrics_uw = predictor.evaluate(X_test_uw, y_test_uw)
        
        # Random Forest Model
        print("\nTraining Random Forest model with weights...")
        predictor.train(X_train_w, y_train_w, model_type='rf')
        
        print("\nMaking Random Forest predictions with weights...")
        rf_predictions_w = predictor.predict(X_test_w)
        
        # Evaluate Random Forest model with weights
        rf_metrics_w = predictor.evaluate(X_test_w, y_test_w)
        
        print("\nTraining Random Forest model without weights...")
        predictor.train(X_train_uw, y_train_uw, model_type='rf')
        
        print("\nMaking Random Forest predictions without weights...")
        rf_predictions_uw = predictor.predict(X_test_uw)
        
        # Evaluate Random Forest model without weights
        rf_metrics_uw = predictor.evaluate(X_test_uw, y_test_uw)
        
        # Print evaluation metrics
        print("\nGradient Boosting Model with Weights - Testing Metrics:")
        for metric in test_metrics_w:
            print(f"Day {metric['day']}: RMSE = {metric['rmse']:.4f}, R² = {metric['r2']:.4f}, MAPE = {metric['mape']:.4f}")
        
        print("\nGradient Boosting Model without Weights - Testing Metrics:")
        for metric in test_metrics_uw:
            print(f"Day {metric['day']}: RMSE = {metric['rmse']:.4f}, R² = {metric['r2']:.4f}, MAPE = {metric['mape']:.4f}")
        
        print("\nRandom Forest Model with Weights - Testing Metrics:")
        for metric in rf_metrics_w:
            print(f"Day {metric['day']}: RMSE = {metric['rmse']:.4f}, R² = {metric['r2']:.4f}, MAPE = {metric['mape']:.4f}")
        
        print("\nRandom Forest Model without Weights - Testing Metrics:")
        for metric in rf_metrics_uw:
            print(f"Day {metric['day']}: RMSE = {metric['rmse']:.4f}, R² = {metric['r2']:.4f}, MAPE = {metric['mape']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in price prediction pipeline: {e}")
        raise

if __name__ == "__main__":
    run_price_prediction()
