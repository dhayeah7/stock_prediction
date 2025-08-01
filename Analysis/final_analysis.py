import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime as dt
import seaborn as sns

# Function to load and preprocess tweet sentiment data
def load_tweet_data(file_path, symbol):
    """
    Load tweet sentiment data from CSV file and prepare it for analysis
    
    Parameters:
    file_path (str): Path to CSV file with tweet sentiment data
    symbol (str): Stock symbol to filter tweets
    
    Returns:
    DataFrame: Processed tweet sentiment data
    """
    try:
        tweet_data = pd.read_csv(file_path)
        tweet_data['date'] = pd.to_datetime(tweet_data['date'])
        
        # Filter for symbol if provided
        if symbol:
            tweet_data = tweet_data[tweet_data['symbol'].str.upper() == symbol.upper()]
        
        # Aggregate by date with better handling of timestamps
        daily_sentiment = tweet_data.groupby(tweet_data['date'].dt.date).agg({
            'sentiment_score': ['mean', 'count', 'std'],
            'source': lambda x: x.mode()[0] if not x.empty else None
        }).reset_index()
        
        # Flatten multi-level columns
        daily_sentiment.columns = ['date', 'avg_sentiment', 'tweet_count', 'sentiment_std', 'source']
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        return daily_sentiment
    except Exception as e:
        print(f"Error loading tweet data: {e}")
        return None

# Function to load stock data
def load_stock_data(symbol, start_date, end_date):
    """
    Load historical stock data using yfinance
    
    Parameters:
    symbol (str): Stock symbol
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    DataFrame: Historical stock data
    """
    # Set auto_adjust explicitly to avoid warning
    stock_data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    
    # Reset index to convert Date from index to column
    stock_data.reset_index(inplace=True)
    
    # Fix for multi-level columns - flatten if needed
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in stock_data.columns]
    
    return stock_data

# Function to merge stock and sentiment data
def merge_data(stock_data, sentiment_data):
    """
    Merge stock price data with sentiment data
    
    Parameters:
    stock_data (DataFrame): Historical stock data
    sentiment_data (DataFrame): Sentiment data aggregated by date
    
    Returns:
    DataFrame: Merged dataframe with stock and sentiment data
    """
    # Print column names to debug
    print("Stock data columns:", stock_data.columns.tolist())
    print("Sentiment data columns:", sentiment_data.columns.tolist())
    
    # Merge datasets on date
    merged_data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date', how='left')
    
    # Fill missing sentiment data with neutral values
    merged_data['avg_sentiment'] = merged_data['avg_sentiment'].fillna(0)
    merged_data['tweet_count'] = merged_data['tweet_count'].fillna(0)
    merged_data['sentiment_std'] = merged_data['sentiment_std'].fillna(0)
    
    # Drop redundant date column
    merged_data.drop('date', axis=1, inplace=True)
    
    return merged_data

# Function to prepare data for LSTM model
def prepare_lstm_data(data, lookback_days=30, prediction_days=1, test_split=0.2):
    """
    Prepare data for LSTM model with a sliding window approach
    
    Parameters:
    data (DataFrame): Combined stock and sentiment data
    lookback_days (int): Number of previous days to use for prediction
    prediction_days (int): Number of days to predict ahead
    test_split (float): Proportion of data to use for testing
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler objects
    """
    # Select features - ensure these columns exist in your dataframe
    features = ['Close', 'Volume', 'avg_sentiment', 'tweet_count', 'sentiment_std']
    
    # Make sure all required columns exist
    for feature in features:
        if feature not in data.columns:
            # Check for adjusted column names
            if feature == 'Close' and 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
            elif feature not in ['avg_sentiment', 'tweet_count', 'sentiment_std']:
                raise ValueError(f"Required column {feature} not found in data")
    
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale the Close price separately (for target)
    price_scaled = price_scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    # Scale the features
    features_scaled = feature_scaler.fit_transform(data[features].values)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(lookback_days, len(data) - prediction_days + 1):
        X.append(features_scaled[i-lookback_days:i])
        y.append(price_scaled[i + prediction_days - 1])
    
    X, y = np.array(X), np.array(y)
    
    # Split data into train and test sets
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, price_scaler, feature_scaler, features

# Function to build and train LSTM model
def build_lstm_model(X_train, y_train, X_test, y_test):
    """
    Build and train LSTM model
    
    Parameters:
    X_train, y_train, X_test, y_test: Training and testing data
    
    Returns:
    model: Trained LSTM model
    history: Training history
    """
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Reduced from 100 to make it run faster
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return model, history

# Function to make predictions and evaluate the model
def evaluate_model(model, X_test, y_test, price_scaler):
    """
    Evaluate model performance and make predictions
    
    Parameters:
    model: Trained LSTM model
    X_test, y_test: Test data
    price_scaler: Scaler used for the price data
    
    Returns:
    DataFrame: Predictions compared to actual prices
    float: RMSE of the model
    """
    # Make predictions
    predicted_prices_scaled = model.predict(X_test)
    
    # Inverse transform to get actual prices
    predicted_prices = price_scaler.inverse_transform(predicted_prices_scaled)
    actual_prices = price_scaler.inverse_transform(y_test)
    
    # Calculate errors
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    print(f'Root Mean Square Error: {rmse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Actual': actual_prices.flatten(),
        'Predicted': predicted_prices.flatten()
    })
    
    return results, rmse

# Function to predict next day price and investment signal
def predict_next_day(model, last_sequence, price_scaler, feature_scaler, features, data):
    """
    Predict the next day's price and provide investment signal
    
    Parameters:
    model: Trained LSTM model
    last_sequence: Last available sequence of data
    price_scaler: Scaler for price data
    feature_scaler: Scaler for feature data
    features: List of feature names
    data: Original data DataFrame
    
    Returns:
    float: Predicted price
    str: Investment signal ('positive', 'negative', 'neutral')
    """
    # Predict next day price
    next_day_price_scaled = model.predict(np.array([last_sequence]))
    next_day_price = price_scaler.inverse_transform(next_day_price_scaled)[0][0]
    
    # Calculate recent price trend (5-day)
    recent_prices = data['Close'].iloc[-5:].values
    recent_avg = np.mean(recent_prices)
    
    # Calculate recent sentiment
    recent_sentiment = data['avg_sentiment'].iloc[-7:].mean()
    
    # Determine investment signal
    price_signal = 1 if next_day_price > data['Close'].iloc[-1] else -1
    
    # Combine price prediction with sentiment for final signal
    if price_signal > 0 and recent_sentiment > 0.1:
        signal = 'positive'  # Strong buy
    elif price_signal > 0 and recent_sentiment >= -0.1:
        signal = 'neutral'   # Hold/weak buy
    elif price_signal < 0 and recent_sentiment < -0.1:
        signal = 'negative'  # Strong sell
    else:
        signal = 'neutral'   # Hold
    
    return next_day_price, signal

# Function to visualize results
def visualize_results(data, results, next_day_prediction, signal):
    """
    Create visualization of historical prices, predictions, and future prediction
    
    Parameters:
    data: Original data DataFrame
    results: DataFrame with prediction results
    next_day_prediction: Predicted price for next day
    signal: Investment signal
    
    Returns:
    plt: matplotlib plot object
    """
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted prices
    plt.subplot(2, 1, 1)
    plt.plot(data['Close'].iloc[-len(results)-30:-len(results)], label='Historical Data')
    plt.plot(range(len(data)-len(results), len(data)), results['Actual'], label='Actual Price')
    plt.plot(range(len(data)-len(results), len(data)), results['Predicted'], label='Predicted Price')
    
    # Add next day prediction
    next_date = data['Date'].iloc[-1] + pd.Timedelta(days=1)
    plt.scatter(len(data), next_day_prediction, color='red', s=100, label=f'Next Day: ${next_day_prediction:.2f}')
    
    # Add signal indicator
    signal_color = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
    plt.axvspan(len(data)-1, len(data)+1, alpha=0.2, color=signal_color[signal])
    
    plt.title(f'Stock Price Prediction with Signal: {signal.upper()}')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    
    # Plot 2: Sentiment analysis and correlation with price
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot sentiment
    ax1.bar(range(len(data[-30:])), data['avg_sentiment'].iloc[-30:], alpha=0.3, color='blue', label='Sentiment')
    ax1.set_ylabel('Sentiment Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot price on secondary axis
    ax2.plot(range(len(data[-30:])), data['Close'].iloc[-30:], color='green', label='Price')
    ax2.set_ylabel('Price ($)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title('Sentiment Analysis vs Stock Price (Last 30 Days)')
    plt.xlabel('Days')
    plt.tight_layout()
    
    return plt

# Main function to run the entire pipeline
def run_stock_prediction(stock_symbol, tweet_data_path, prediction_days=1, lookback_days=30):
    """
    Run the entire stock prediction pipeline
    
    Parameters:
    stock_symbol (str): Stock symbol to analyze
    tweet_data_path (str): Path to tweet sentiment CSV file
    prediction_days (int): Number of days to predict ahead
    lookback_days (int): Number of previous days to use for prediction
    
    Returns:
    tuple: Prediction results and visualization
    """
    # Calculate dates for data fetching - FIXED: Define these variables first
    end_date = dt.datetime.now()
    # Get extra data to account for weekends and holidays
    start_date = end_date - dt.timedelta(days=lookback_days + 30)
    
    print(f"Analyzing {stock_symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Load data
    tweet_data = load_tweet_data(tweet_data_path, stock_symbol)
    stock_data = load_stock_data(stock_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Debug: print the first few rows of each dataframe
    print("\nFirst few rows of tweet data:")
    print(tweet_data.head())
    print("\nFirst few rows of stock data:")
    print(stock_data.head())
    
    # Merge data
    combined_data = merge_data(stock_data, tweet_data)
    # Rename columns to match expected names
    combined_data.rename(columns={
        'Close_MSFT': 'Close',
        'Volume_MSFT': 'Volume'
    }, inplace=True)
    # Debug: print the first few rows of combined data
    print("\nFirst few rows of combined data:")
    print(combined_data.head())
    
    # Prepare data for LSTM
    X_train, X_test, y_train, y_test, price_scaler, feature_scaler, features = prepare_lstm_data(
        combined_data, lookback_days, prediction_days
    )
    
    print(f"\nTraining LSTM model with {len(X_train)} training samples and {len(X_test)} test samples")
    
    # Build and train model
    model, history = build_lstm_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results, rmse = evaluate_model(model, X_test, y_test, price_scaler)
    
    # Prepare last sequence for next day prediction
    last_sequence = X_test[-1:][0]
    
    # Predict next day
    next_day_price, signal = predict_next_day(
        model, last_sequence, price_scaler, feature_scaler, features, combined_data
    )
    
    print(f"Predicted price for next trading day: ${next_day_price:.2f}")
    print(f"Investment signal: {signal.upper()}")
    
    # Visualize results
    plot = visualize_results(combined_data, results, next_day_price, signal)
    
    return next_day_price, signal, plot, model

# Example usage
if __name__ == "__main__":
    # Replace these with your actual values
    STOCK_SYMBOL = "MSFT"  # Microsoft as in your example
    TWEET_DATA_PATH = "merged_news_stocks1.csv"  # Update with your actual file path
    
    try:
        next_price, signal, plot, model = run_stock_prediction(
            stock_symbol=STOCK_SYMBOL,
            tweet_data_path=TWEET_DATA_PATH,
            prediction_days=1,
            lookback_days=30
        )
        
        plt.savefig('stock_prediction.png')  # Save the figure to a file
        plt.show()
        
        print(f"Analysis complete. Results saved to stock_prediction.png")
        
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()