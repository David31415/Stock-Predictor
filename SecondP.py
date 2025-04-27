import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import argparse
import time

def get_stock_data(ticker, period='1y'):
    """Fetch historical stock data for analysis"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if len(hist) < 30:  # Ensure sufficient data
            return None
        return hist
    except Exception as e:
        # Silent failure - will be handled by the caller
        return None

def calculate_features(df):
    """Calculate technical indicators for prediction"""
    if df is None or len(df) < 30:
        return None
    
    # Create a copy to avoid SettingWithCopyWarning
    data = df.copy()
    
    # Add technical indicators
    # Moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    # Volume indicators
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
    
    # Price momentum
    data['Price_Change'] = data['Close'].pct_change()
    data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
    
    # Volatility
    data['Volatility'] = data['Close'].rolling(window=10).std()
    
    # Relative strength
    data['RSI'] = calculate_rsi(data['Close'], 14)
    
    # Fill NaN values created by rolling calculations
    data = data.dropna()
    
    return data

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    delta = delta[1:]  # Remove first NaN

    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    
    # Handle division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=prices.index[1:])

def prepare_training_data(data, prediction_days=5):
    """Prepare features and target for prediction model"""
    if data is None or len(data) < prediction_days + 10:
        return None, None
    
    features = data[['Close', 'Volume', 'MA5', 'MA10', 'MA20',
                     'Volume_Change', 'Volume_MA5', 'Price_Change',
                     'Price_Change_5d', 'Volatility', 'RSI']].copy()
    
    # Target: price percentage change after prediction_days
    data['Target'] = data['Close'].shift(-prediction_days) / data['Close'] - 1
    target = data['Target'][:-prediction_days]  # Remove last rows that have NaN targets
    features = features[:-prediction_days]  # Match features to target
    
    return features, target

def train_model(features, target):
    """Train RandomForest regression model"""
    if features is None or target is None:
        return None, None
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(scaled_features, target)
    
    return model, scaler

def predict_stock_movement(ticker, model, scaler):
    """Predict future price movement for a stock"""
    try:
        # Get latest data
        stock_data = get_stock_data(ticker, period='1y')
        if stock_data is None or len(stock_data) < 30:
            return None, None, None, None
        
        # Process features
        processed_data = calculate_features(stock_data)
        if processed_data is None:
            return None, None, None, None
        
        # Get latest price
        current_price = processed_data['Close'].iloc[-1]
        
        # Get latest features for prediction
        latest_features = processed_data[['Close', 'Volume', 'MA5', 'MA10', 'MA20',
                                         'Volume_Change', 'Volume_MA5', 'Price_Change',
                                         'Price_Change_5d', 'Volatility', 'RSI']].iloc[-1:]
        
        # Scale and predict
        scaled_features = scaler.transform(latest_features)
        predicted_change = model.predict(scaled_features)[0]
        
        # Calculate predicted price
        predicted_price = current_price * (1 + predicted_change)
        
        return ticker, current_price, predicted_price, predicted_change
    except Exception as e:
        return None, None, None, None

def is_valid_stock(ticker):
    """Check if the ticker is valid and active"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Check if we can get basic info
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return True
        return False
    except Exception:
        return False

def get_current_price(ticker):
    """Get the current price of a stock safely"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if len(data) > 0 and 'Close' in data:
            return data['Close'].iloc[-1]
        return None
    except Exception:
        return None

def find_promising_stocks(price_min, price_max, min_gain):
    """Find stocks within price range with predicted gains above threshold"""
    price_range = (price_min, price_max)
    
    # Get list of stocks to analyze
    # For demonstration, we'll use a sample list (in reality, you'd get this from an API or database)
    stock_universe = get_stock_universe_in_price_range(price_range)
    print(f"Found {len(stock_universe)} stocks in price range ${price_range[0]}-${price_range[1]}")
    
    if len(stock_universe) == 0:
        print("No stocks found in the specified price range.")
        return pd.DataFrame()
    
    # Train model using a representative stock (e.g., SPY)
    print("Training prediction model...")
    sp500_data = get_stock_data('SPY', period='2y')
    processed_sp500 = calculate_features(sp500_data)
    features, target = prepare_training_data(processed_sp500)
    model, scaler = train_model(features, target)
    
    if model is None or scaler is None:
        print("Error: Could not train prediction model. Check data availability for SPY.")
        return pd.DataFrame()
    
    # Analyze each stock
    results = []
    skipped = 0
    for i, ticker in enumerate(stock_universe):
        if i % 10 == 0:
            print(f"Analyzing stocks... {i+1}/{len(stock_universe)}")
            
        ticker_data = predict_stock_movement(ticker, model, scaler)
        ticker_name, current_price, predicted_price, predicted_change = ticker_data
        
        if ticker_name and current_price and predicted_price and predicted_change:
            results.append({
                'Ticker': ticker_name,
                'Current Price': current_price,
                'Predicted Price': predicted_price,
                'Predicted Change %': predicted_change * 100,
                'Meets Threshold': predicted_change >= min_gain
            })
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Skipped {skipped} stocks due to insufficient data or other issues.")
    
    # Create DataFrame and filter
    if not results:
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    promising_stocks = results_df[results_df['Meets Threshold'] == True].sort_values(
        by='Predicted Change %', ascending=False
    )
    
    return promising_stocks

def get_stock_universe_in_price_range(price_range):
    """
    Get list of stocks in the specified price range
    This is a simplified version - in a real application, you'd need to:
    1. Fetch all stock symbols from a database or API
    2. Filter by current price
    """
    # Updated sample universe - removed HEXO and added more current stocks
    sample_universe = [
        # Small-cap stocks examples
        'SNDL', 'ACB', 'TXMD', 'GNPX', 'CTRM', 'IDEX', 'SENS', 'NNDM',
        # Mid-cap stocks examples
        'PLUG', 'BLNK', 'MARA', 'RIOT', 'MVIS', 'GSAT', 'WKHS', 'GOEV',
        # Some energy stocks
        'PTEN', 'RIG', 'NOG', 'CPE', 'SM', 'OXY', 'MRO', 'CLF', 'BTU',
        # Some healthcare stocks
        'OCGN', 'INO', 'SRNE', 'ATOS', 'NTLA', 'EDIT', 'CRSP',
        # Tech stocks
        'BB', 'NOK', 'WISH', 'SKLZ', 'PLTR', 'SOFI',
        # Additional stocks across various price ranges
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'F', 'GE', 'BAC', 'WFC', 'T', 'VZ',
        'XOM', 'CVX', 'PFE', 'JNJ', 'KO', 'PEP', 'WMT', 'TGT', 'AMD', 'INTC', 'NVDA',
        # Added more stocks that might be in various price ranges
        'UBER', 'LYFT', 'SNAP', 'PINS', 'TWTR', 'HOOD', 'PTON', 'ZM', 'DASH', 'ABNB',
        'COIN', 'RBLX', 'U', 'BROS', 'DKNG', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI'
    ]
    
    in_range_stocks = []
    errors = []
    
    print("Screening stocks for price range...")
    for i, ticker in enumerate(sample_universe):
        if i % 20 == 0 and i > 0:
            print(f"Checked {i}/{len(sample_universe)} tickers...")
            # Add a small delay to avoid hitting rate limits
            time.sleep(1)
            
        price = get_current_price(ticker)
        if price is not None and price_range[0] <= price <= price_range[1]:
            in_range_stocks.append(ticker)
    
    # Report any errors but continue with valid stocks
    if errors:
        print(f"Note: {len(errors)} stocks were skipped due to data issues.")
    
    return in_range_stocks

def main():
    """Main function to run the analysis with command line arguments"""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Stock Price Prediction Tool')
    parser.add_argument('--min-price', type=float, default=5.0, help='Minimum stock price (default: 5.0)')
    parser.add_argument('--max-price', type=float, default=15.0, help='Maximum stock price (default: 15.0)')
    parser.add_argument('--min-roi', type=float, default=0.10, help='Minimum ROI percentage as decimal (default: 0.10 for 10%%)')
    args = parser.parse_args()
    
    # Get parameters from command line
    price_min = args.min_price
    price_max = args.max_price
    min_gain = args.min_roi
    
    print("Stock Price Prediction Tool")
    print(f"Analyzing stocks in ${price_min}-${price_max} price range with {min_gain*100}%+ potential growth")
    print("=" * 70)
    
    # Find promising stocks
    promising_stocks = find_promising_stocks(price_min, price_max, min_gain)
    
    # Display results
    print("\nResults:")
    print("=" * 70)
    if len(promising_stocks) > 0:
        print(f"Found {len(promising_stocks)} stocks with predicted gains of {min_gain*100}% or more:")
        print(promising_stocks[['Ticker', 'Current Price', 'Predicted Price', 'Predicted Change %']])
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_predictions_{timestamp}.csv"
        promising_stocks.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
    else:
        print("No stocks found meeting the criteria.")
    
    print("\nDisclaimer: These predictions are based on historical patterns and technical analysis.")
    print("They should not be considered financial advice. Always do your own research before investing.")

def interactive_mode():
    """Run the program in interactive mode, prompting user for inputs"""
    print("Stock Price Prediction Tool - Interactive Mode")
    print("=" * 70)
    
    # Get user inputs
    while True:
        try:
            min_price = float(input("Enter minimum stock price: $"))
            if min_price <= 0:
                print("Price must be greater than 0. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    while True:
        try:
            max_price = float(input("Enter maximum stock price: $"))
            if max_price <= min_price:
                print(f"Maximum price must be greater than minimum price (${min_price}). Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    while True:
        try:
            min_roi_percent = float(input("Enter minimum ROI percentage: "))
            if min_roi_percent <= 0:
                print("ROI percentage must be greater than 0. Please try again.")
                continue
            min_roi = min_roi_percent / 100  # Convert percentage to decimal
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"\nAnalyzing stocks in ${min_price}-${max_price} price range with {min_roi*100}%+ potential growth")
    print("=" * 70)
    
    # Find promising stocks
    promising_stocks = find_promising_stocks(min_price, max_price, min_roi)
    
    # Display results
    print("\nResults:")
    print("=" * 70)
    if len(promising_stocks) > 0:
        print(f"Found {len(promising_stocks)} stocks with predicted gains of {min_roi*100}% or more:")
        print(promising_stocks[['Ticker', 'Current Price', 'Predicted Price', 'Predicted Change %']])
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_predictions_{timestamp}.csv"
        promising_stocks.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
    else:
        print("No stocks found meeting the criteria.")
    
    print("\nDisclaimer: These predictions are based on historical patterns and technical analysis.")
    print("They should not be considered financial advice. Always do your own research before investing.")

if __name__ == "__main__":
    # Check if any command line arguments were provided
    import sys
    if len(sys.argv) > 1:
        main()  # Run with command line arguments
    else:
        interactive_mode()  # Run in interactive mode