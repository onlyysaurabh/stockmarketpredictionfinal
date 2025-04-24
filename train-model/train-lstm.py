# train-lstm.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report, recall_score
import os
import sys
import datetime
import pymongo
from pymongo import MongoClient
import argparse
import shap  # Import SHAP for model explainability
import matplotlib.pyplot as plt
import tensorflow as tf

# Add parent directory to path to import mongo_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stocks.mongo_utils import get_mongo_db, STOCK_PRICES_COLLECTION

# --- Parse Command Line Arguments ---
def parse_args():
    """Parse command line arguments for training parameters."""
    parser = argparse.ArgumentParser(description='Train LSTM model for stock price prediction')
    
    # Stock selection
    stock_group = parser.add_mutually_exclusive_group()
    stock_group.add_argument('--symbol', type=str, help='Single stock symbol to train on (e.g., AAPL)')
    stock_group.add_argument('--symbols', type=str, help='Comma-separated list of stock symbols to train on (e.g., AAPL,MSFT,GOOGL)')
    
    # Date range
    parser.add_argument('--start-date', type=str, default="2020-02-25", 
                      help='Start date for training data (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str, default="2025-02-25",
                      help='End date for training data (YYYY-MM-DD format)')
    
    # Model parameters
    parser.add_argument('--seq-length', type=int, default=60,
                      help='Sequence length for LSTM input (default: 60)')
    parser.add_argument('--lstm-units', type=int, default=60,
                      help='Number of LSTM units (default: 60)')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                      help='Dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=75,
                      help='Number of training epochs (default: 75)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size (default: 32)')
    parser.add_argument('--patience', type=int, default=15,
                      help='Early stopping patience (default: 15)')
                      
    # Window parameters
    parser.add_argument('--short-window', type=int, default=20,
                      help='Short moving average window (default: 20)')
    parser.add_argument('--long-window', type=int, default=50,
                      help='Long moving average window (default: 50)')
    parser.add_argument('--lag-days', type=int, default=1,
                      help='Number of days for lagged features (default: 1)')
                      
    return parser.parse_args()

# --- Helper function to get stock symbols ---
def get_stock_symbols(args):
    """Get stock symbols from arguments or CSV file."""
    if args.symbol:
        return [args.symbol.upper()]
    elif args.symbols:
        return [symbol.strip().upper() for symbol in args.symbols.split(',')]
    else:
        # Default: load from CSV
        stocks_file = "stock_symbols.csv"
        try:
            stocks_df = pd.read_csv(stocks_file)
            return stocks_df['Symbol'].tolist()
        except FileNotFoundError:
            print(f"Error: {stocks_file} not found.")
            sys.exit(1)
        except KeyError:
            print(f"Error: 'Symbol' column not found in {stocks_file}.")
            sys.exit(1)

def load_data_from_mongodb(mongo_uri, db_name, stock_symbol, 
                          start_date=None, end_date=None):
    """Load stock data from MongoDB using the correct collection and data structure"""
    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db['stock_prices']  # Use 'stock_prices' collection directly

        # Find the single document for the stock symbol
        stock_doc = collection.find_one({"symbol": stock_symbol})

        if not stock_doc:
            raise ValueError(
                f"No document found for symbol '{stock_symbol}' in MongoDB "
                f"collection 'stock_prices'."
            )

        historical_data = stock_doc.get("historical_data", [])
        if not historical_data:
             raise ValueError(
                f"No 'historical_data' found for symbol '{stock_symbol}'."
            )

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)

        if 'Date' not in df.columns:
             raise ValueError(
                f"Required field 'Date' not found in 'historical_data'."
            )
            
        # Ensure 'Date' is datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter by date range if provided
        if start_date and end_date:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]

        if df.empty:
            raise ValueError(
                f"No historical data found for symbol '{stock_symbol}' "
                f"within the specified date range ({start_date} to {end_date})."
            )

        # Check for required columns after filtering
        required_cols = ['Date', 'Close', 'Volume', 'Open', 'High', 'Low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Check if columns are in different case (MongoDB data has different case)
            column_map = {
                'date': 'Date',
                'close': 'Close', 
                'volume': 'Volume',
                'open': 'Open',
                'high': 'High',
                'low': 'Low'
            }
            df.rename(columns=column_map, inplace=True, errors='ignore')
            
            # Check again for missing columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in historical_data: {', '.join(missing_cols)}"
                )
        
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)  # Ensure data is sorted by date
        
        return df
    
    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        raise Exception(f"Error loading data from MongoDB: {e}")
    finally:
        if client:
            client.close()

def calculate_technical_indicators(df, short_window=20, long_window=50, lag_days=1):
    """
    Calculates technical indicators: Moving Averages, and Lagged Returns.

    Args:
        df: DataFrame with 'Close' prices.
        short_window: Window size for the short moving average.
        long_window: Window size for the long moving average.
        lag_days: Number of days to lag returns.

    Returns:
        DataFrame with added indicator columns. Returns an empty DataFrame on error.
    """
    if 'Close' not in df.columns:
        print("Error: 'Close' column not found in DataFrame.")
        return pd.DataFrame()

    df = df.copy()  # Work on a copy to avoid modifying the original

    # Simple Moving Averages
    df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()

    # Lagged Returns (for creating target variable and features)
    df[f'Lagged_Return_{lag_days}'] = df['Close'].shift(lag_days)

    # Drop rows with NaN values introduced by indicators
    df.dropna(inplace=True)
    return df



def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM input.

    Args:
        data: NumPy array of features.
        seq_length: Length of the input sequences.

    Returns:
        X: Input sequences.
        y: Corresponding target values (next day's close).
    """
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])  # Assuming 'Close' is the first column after scaling
    return np.array(X), np.array(y)


def create_classification_labels(y_regression, y_lagged):
    """
    Creates classification labels (Up/Down) based on regression targets.
    """
    return np.where(y_regression > y_lagged, 1, 0)  # 1 for Up, 0 for Down


def evaluate_regression(y_true, y_pred):
    """Evaluates regression performance and calculates MAPE."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Calculate MAPE, handling potential division by zero
    mape_numerator = np.abs(y_true - y_pred)
    mape_denominator = np.abs(y_true)
    # Avoid division by zero for zero values in y_true, replace with small value or filter out
    mape_denominator = np.where(mape_denominator == 0, np.finfo(float).eps, mape_denominator) # Replace 0 with epsilon
    mape = np.mean(mape_numerator / mape_denominator) * 100

    print("\nRegression Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%") # Added MAPE
    print(f"  R-squared (R2): {r2:.4f}")
    return mse, rmse, mae, r2, mape # Return MAPE


def evaluate_classification(y_true, y_pred):
    """Evaluates classification performance and calculates additional metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, zero_division=0) # added zero_division=0 to handle cases with no predicted samples for a class

    if conf_matrix.size == 4: # Standard 2x2 confusion matrix
        tn, fp, fn, tp = conf_matrix.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else: # Handle cases with less than 2 classes in predictions
        print("Warning: Confusion matrix is not 2x2, possibly due to single class in y_true or y_pred.")
        tn, fp, fn, tp = 0, 0, 0, 0 # Set metrics to zero or handle as needed
        sensitivity, specificity, precision, f1 = 0, 0, 0, 0


    print("\nClassification Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print("  Confusion Matrix:\n", conf_matrix)
    print(f"  Sensitivity (Recall or True Positive Rate): {sensitivity:.2f}") # Added Sensitivity
    print(f"  Specificity (True Negative Rate): {specificity:.2f}")      # Added Specificity
    print(f"  Precision: {precision:.2f}")          # Added Precision
    print(f"  F1-Score: {f1:.2f}")              # Added F1-Score
    print("  Classification Report:\n", class_report)
    return accuracy, conf_matrix, class_report, sensitivity, specificity, precision, f1 # Return additional metrics


def generate_lstm_shap_values(model, X_test, feature_names):
    """Generate SHAP values for LSTM model explanation"""
    print("\nGenerating SHAP values for LSTM model explanation...")
    feature_importance_dict = {}
    
    try:
        # Create a function that returns model predictions
        def model_predict(x):
            # Convert pandas DataFrame to numpy array with correct shape for LSTM
            x_array = np.array(x)
            # Reshape to (samples, timesteps, features)
            x_reshaped = x_array.reshape((x_array.shape[0], X_test.shape[1], X_test.shape[2]))
            return model.predict(x_reshaped)
        
        # Use a subset of test data for performance reasons
        max_samples = min(100, X_test.shape[0])
        X_sample = X_test[:max_samples]
        
        # Flatten the 3D input to 2D for SHAP explainer
        X_sample_2d = X_sample.reshape(X_sample.shape[0], -1)
        
        # Create a background dataset for the explainer (small subset of data)
        background_data = X_sample_2d[:min(20, X_sample_2d.shape[0])]
        
        # Create SHAP explainer
        explainer = shap.DeepExplainer(model, X_sample.reshape(X_sample.shape[0], X_sample.shape[1], X_sample.shape[2]))
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample.reshape(X_sample.shape[0], X_sample.shape[1], X_sample.shape[2]))
        
        # Calculate feature importance from SHAP values (average absolute SHAP values across samples)
        # For sequential data, we need to consider the time dimension
        shap_importance = np.abs(shap_values[0]).mean(axis=0)  # Average across samples
        
        # Sum importance across time steps for each feature
        feature_importance = shap_importance.sum(axis=0)  # Sum across time steps
        
        # Create feature importance dictionary
        for i, importance in enumerate(feature_importance):
            if i < len(feature_names):
                feature_importance_dict[feature_names[i]] = float(importance)
        
        # Print top features by importance
        print("\nLSTM SHAP Feature Importance (Top 10):")
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")
        
    except Exception as e:
        print(f"Error generating SHAP values for LSTM model: {e}")
    
    return feature_importance_dict


def train_lstm_model(symbol, start_date, end_date, evaluation_results_collection, seq_length=60, short_window=20, long_window=50, lag_days=1, lstm_units=50, dropout_rate=0.2, epochs=50, batch_size=32, patience=10):
    """
    Trains and evaluates LSTM models for regression and classification and stores results in MongoDB.

    Args:
        symbol: Stock symbol (e.g., "AAPL").
        start_date: Start date for data fetching (YYYY-MM-DD).
        end_date: End date for data fetching (YYYY-MM-DD).
        evaluation_results_collection: MongoDB collection to store results.
        seq_length: Sequence length for LSTM input.
        short_window:  Short moving average window.
        long_window: Long moving average window.
        lag_days: Number of days to lag returns.
        lstm_units: Number of LSTM units.
        dropout_rate: Dropout rate.
        epochs: Number of training epochs.
        batch_size: Batch size.
        patience:  Patience for EarlyStopping

    Returns:
        evaluation_metrics (dict): Dictionary containing evaluation metrics.
        None if data download or processing fails.
    """

    # --- 1. Data Fetching and Preprocessing ---
    try:
        # Use environment variables for MongoDB connection details
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        db_name = os.getenv('MONGO_DB_NAME', 'stock_prices')
        data = load_data_from_mongodb(mongo_uri, db_name, symbol, start_date, end_date)
        
        if data is None or data.empty:
            print(f"No data found for {symbol} between {start_date} and {end_date}.")
            return None # Indicate failure
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None # Indicate failure

    data = calculate_technical_indicators(data, short_window, long_window, lag_days)
    if data.empty: # check if indicators were calculated correctly
        return None # Indicate failure

    # Feature Selection and Scaling
    features = ['Close', 'SMA_Short', 'SMA_Long', 'EMA_Short', 'EMA_Long', f'Lagged_Return_{lag_days}']  # Include indicators
    data_filtered = data[features]


    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_filtered)


    # --- 2. Create Sequences and Labels ---
    X, y_regression = create_sequences(scaled_data, seq_length)
    y_classification = create_classification_labels(y_regression, data_filtered[f'Lagged_Return_{lag_days}'].values[seq_length:])

    # --- 3. Train/Test Split ---
    # Split based on time to preserve chronological order
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train_reg, y_test_reg = y_regression[:train_size], y_regression[train_size:]
    y_train_class, y_test_class = y_classification[:train_size], y_classification[train_size:]


    # --- 4. Build LSTM Regression Model ---

    # Regression Model
    model_reg = Sequential([
        LSTM(lstm_units, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(lstm_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])
    model_reg.compile(optimizer='adam', loss='mse')

    # --- 5. Train Regression Model with Callbacks ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Use relative path instead of absolute path
    MODEL_SAVE_DIR_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train-model')
    model_dir = os.path.join(MODEL_SAVE_DIR_BASE, symbol, f"lstm-{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Callbacks for early stopping and saving best models
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint_reg = ModelCheckpoint(os.path.join(model_dir, "model.h5"), monitor='val_loss', save_best_only=True)

    print(f"Training Regression Model for {symbol}...")
    history_reg = model_reg.fit(X_train, y_train_reg, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping, checkpoint_reg], verbose=0) # Reduced verbosity


    # --- 6. Evaluate Models ---
    print(f"\n--- Evaluation for {symbol} ---")
    # Regression Evaluation
    model_reg.load_weights(os.path.join(model_dir, "model.h5")) # Load best weights
    y_pred_reg = model_reg.predict(X_test, verbose=0) # Reduced verbosity during prediction
    # Inverse transform to get original scale
    y_pred_reg_orig = scaler.inverse_transform(np.hstack([y_pred_reg, np.zeros((y_pred_reg.shape[0], scaled_data.shape[1]-1))]))[:, 0]
    y_test_reg_orig = scaler.inverse_transform(np.hstack([y_test_reg.reshape(-1,1), np.zeros((y_test_reg.shape[0], scaled_data.shape[1]-1))]))[:, 0]
    mse, rmse, mae, r2, mape = evaluate_regression(y_test_reg_orig, y_pred_reg_orig) # Get MAPE

    # Classification Evaluation from Regression Model
    # Get lagged prices for the test set corresponding to the regression predictions
    y_lagged_test = data_filtered[f'Lagged_Return_{lag_days}'].values[train_size + seq_length:]  # Adjusted index to align with test set and sequence length
    y_pred_class_from_reg = create_classification_labels(y_pred_reg_orig, y_lagged_test)


    accuracy, conf_matrix, class_report, sensitivity, specificity, precision, f1 = evaluate_classification(y_test_class, y_pred_class_from_reg) # Evaluate classification metrics

    # Generate SHAP values for feature importance
    feature_names = features
    shap_importance = generate_lstm_shap_values(model_reg, X_test, feature_names)

    print(f"\n--- Model Evaluation Summary for {symbol} ---")
    print(f"Regression Model Metrics:")
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R-squared: {r2:.4f}") # Summarized Regression metrics
    print(f"Classification Metrics (from Regression):")
    print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, F1-Score: {f1:.2f}") # Summarized Classification metrics
    print("\nConfusion Matrix (Classification from Regression):")
    print(conf_matrix) # Print Confusion Matrix

    # --- 7. Save Regression Model (already handled by ModelCheckpoint callback) ---
    print(f"\nRegression model saved to: {os.path.join(model_dir, 'model.h5')}") # Inform user about save path
    
    # Save the scaler and features list for future inference
    try:
        import pickle
        # Save the scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save the features list
        features_path = os.path.join(model_dir, "selected_features.pkl")
        with open(features_path, 'wb') as file:
            pickle.dump(features, file)
        print(f"Feature list saved to: {features_path}")
    except Exception as e:
        print(f"Error saving scaler or feature list: {e}")

    # --- 8. Store Evaluation Results in MongoDB ---
    evaluation_metrics = {
        "stock_symbol": symbol,
        "start_date": start_date, # Keep as string for simplicity
        "end_date": end_date,      # Keep as string for simplicity
        "seq_length": seq_length,
        "short_window": short_window,
        "long_window": long_window,
        "lag_days": lag_days,
        "lstm_units": lstm_units,
        "dropout_rate": dropout_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "classification_confusion_matrix": conf_matrix.tolist(), # Store confusion matrix as list
        "classification_report": class_report, # Store classification report as string
        "shap_feature_importance": shap_importance  # Store SHAP feature importance
    }

    try:
        evaluation_results_collection.insert_one(evaluation_metrics)
        print(f"\nEvaluation results for {symbol} stored in MongoDB.")
    except Exception as e:
        print(f"Error storing evaluation results in MongoDB: {e}")

    return evaluation_metrics # Return the metrics for potential further use



# --- Main Program ---
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # --- MongoDB Connection Details ---
    # Use environment variables for MongoDB connection details
    import os
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGO_DB_NAME', 'stock_prices')
    evaluation_collection_name = "lstm_evaluation_results"

    # --- Date Range for Training and Prediction ---
    start_date_str = args.start_date
    end_date_str = args.end_date
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Please use YYYY-MM-DD format for dates.")
        sys.exit(1)
        
    print(f"Training with date range: {start_date} to {end_date}")

    # --- LSTM Model Parameters ---
    lstm_units = args.lstm_units
    dropout_rate = args.dropout_rate
    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience
    seq_length = args.seq_length
    short_window = args.short_window
    long_window = args.long_window
    lag_days = args.lag_days
    
    print(f"LSTM parameters: units={lstm_units}, dropout={dropout_rate}, epochs={epochs}, batch_size={batch_size}")
    print(f"Sequence length={seq_length}, short_window={short_window}, long_window={long_window}, lag_days={lag_days}")

    # --- Load Stock Symbols ---
    stock_symbols = get_stock_symbols(args)
    print(f"Training LSTM model for {len(stock_symbols)} stock(s): {', '.join(stock_symbols) if len(stock_symbols) < 5 else ', '.join(stock_symbols[:5]) + '...'}")

    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        evaluation_results_collection = db[evaluation_collection_name]

        for symbol in stock_symbols:
            print(f"\n--- Processing Stock: {symbol} ---")
            try:
                evaluation_metrics = train_lstm_model(
                    symbol, start_date, end_date, evaluation_results_collection, 
                    seq_length, short_window, long_window, lag_days, 
                    lstm_units, dropout_rate, epochs, batch_size, patience
                )
                if evaluation_metrics is None:
                    print(f"Skipping to next stock due to errors processing {symbol}.")
                    continue
            except Exception as e:
                print(f"An error occurred while processing {symbol}: {e}")

    except pymongo.errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main program: {e}")
    finally:
        if client:
            client.close()

    print("\n--- LSTM Stock Price Prediction and Evaluation Completed for all symbols. ---")