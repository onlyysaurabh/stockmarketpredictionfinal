# train-arima.py
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Commented out - not used directly in this script version
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (mean_squared_error, mean_absolute_error, accuracy_score,
                             confusion_matrix, precision_score, recall_score, f1_score)
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
import pymongo
from pymongo import MongoClient
import pickle
import os
import sys
import argparse
from tqdm import tqdm  # Import tqdm for progress bar
from datetime import datetime
from itertools import product
import traceback # For detailed error printing

# Add parent directory to path to import mongo_utils (if needed, otherwise remove)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from stocks.mongo_utils import get_mongo_db, STOCK_PRICES_COLLECTION, STOCKS_COLLECTION # Commented: Using direct connection like xgboost

# --- Parse Command Line Arguments ---
def parse_args():
    """Parse command line arguments for training parameters."""
    parser = argparse.ArgumentParser(description='Train ARIMA model for stock price prediction')

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
    # Default to 'Close' as it's more standard after potential renaming
    parser.add_argument('--price-field', type=str, default="Close",
                      help='Price field to predict (default: Close) - Use capital C')
    parser.add_argument('--max-p', type=int, default=3,
                      help='Maximum p value for ARIMA (default: 3)')
    parser.add_argument('--max-q', type=int, default=3,
                      help='Maximum q value for ARIMA (default: 3)')
    parser.add_argument('--auto-diff', action='store_true',
                      help='Automatically determine differencing order')
    parser.add_argument('--diff-order', type=int, default=1,
                      help='Differencing order if auto-diff is False (default: 1)')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Test size as fraction of data (default: 0.2)')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                      help='Skip feature engineering step')

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

# --- Feature Engineering Functions ---
def calculate_moving_average(series, window=10):
    """Calculates the simple moving average."""
    return series.rolling(window=window).mean()

def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index."""
    delta = series.diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Use com instead of span for compatibility with older pandas versions if needed
    roll_up1 = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    roll_down1 = np.abs(down.ewm(com=period - 1, adjust=True, min_periods=period).mean())
    # Handle division by zero
    RS = roll_up1 / roll_down1
    RS = RS.replace([np.inf, -np.inf], np.nan).fillna(0) # Fill NaNs and Infs resulting from division
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Calculates Moving Average Convergence Divergence."""
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_atr(df, period=14):
    """Calculates Average True Range."""
    # Ensure required columns exist after potential renaming
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("Missing 'High', 'Low', or 'Close' columns for ATR calculation.")
    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
    low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
    # Use pandas max method for robustness
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1, skipna=False)
    atr = tr.rolling(window=period).mean()
    return atr


# --- 1. Load Stock Price Data from MongoDB (Adapted from XGBoost) ---
def load_data_from_mongodb(mongo_uri, db_name, stock_symbol, price_field='Close',
                           start_date=None, end_date=None, feature_engineering=True):
    """Loads stock data from MongoDB, with optional date range and feature engineering."""
    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        # Use the collection name proven successful by XGBoost
        collection = db['stock_prices']

        # Find the single document for the stock symbol (case-insensitive find can be slower,
        # but useful if unsure. Let's stick to the XGBoost way assuming symbol is consistent)
        # For robustness, you could try variations if the first fails:
        stock_doc = collection.find_one({"symbol": stock_symbol})
        if not stock_doc:
             # Try uppercase as a fallback (common convention)
             stock_doc = collection.find_one({"symbol": stock_symbol.upper()})
             if not stock_doc:
                raise ValueError(
                    f"No document found for symbol '{stock_symbol}' (or uppercase) in MongoDB "
                    f"collection 'stock_prices'."
                )

        historical_data = stock_doc.get("historical_data", [])
        if not historical_data:
             raise ValueError(
                f"No 'historical_data' found for symbol '{stock_symbol}'."
            )

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)

        # Check for required 'Date' field BEFORE renaming
        if 'Date' not in df.columns and 'date' not in df.columns:
             raise ValueError(
                f"Required field 'Date' (or 'date') not found in 'historical_data'."
            )

        # --- Renaming and Case Handling (from XGBoost) ---
        column_map = {
            'date': 'Date',
            'close': 'Close',
            'volume': 'Volume',
            'open': 'Open',
            'high': 'High',
            'low': 'Low'
        }
        df.rename(columns=column_map, inplace=True, errors='ignore')

        # Ensure 'Date' is datetime AFTER potential rename
        if 'Date' not in df.columns:
             raise ValueError("Column 'Date' not found after renaming.") # Should not happen if check above passed
        df['Date'] = pd.to_datetime(df['Date'])

        # Filter by date range if provided
        if start_date and end_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]

        if df.empty:
            raise ValueError(
                f"No historical data found for symbol '{stock_symbol}' "
                f"within the specified date range ({start_date} to {end_date})."
            )

        # Check for required columns AFTER filtering and renaming
        # ARIMA needs these for features (or just Close if no features)
        required_cols_base = ['Date', 'Close']
        required_cols_features = ['Volume', 'Open', 'High', 'Low']
        required_cols = required_cols_base + (required_cols_features if feature_engineering else [])

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in filtered historical_data: {', '.join(missing_cols)}"
            )

        # Ensure numeric types for calculations (robustness)
        for col in ['Close', 'Volume', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')


        # Set Date as index and sort
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True) # Ensure data is sorted by date

        # --- Feature Engineering (Moved from original ARIMA load function) ---
        if feature_engineering:
            # Use the potentially renamed 'Close' column
            df['MA_10'] = calculate_moving_average(df['Close'], window=10)
            df['RSI'] = calculate_rsi(df['Close'])
            macd_line, signal_line, _ = calculate_macd(df['Close']) # Ignore histogram if not used
            df['MACD_Line'] = macd_line
            df['MACD_Signal'] = signal_line
            df['ATR'] = calculate_atr(df) # Pass the whole df now
            df['Price_Range'] = df['High'] - df['Low']
            # Ensure Volume is numeric before diff
            df['Volume_Change'] = df['Volume'].diff()

            # Drop NaNs introduced by feature engineering or type coercion
            df.dropna(inplace=True)

            if df.empty:
                raise ValueError(
                    f"DataFrame became empty after feature engineering and dropping NaNs for '{stock_symbol}'."
                )

        # --- Select the target price series ---
        # Ensure the price_field exists after potential renaming and feature engineering
        if price_field not in df.columns:
             raise ValueError(f"Specified price_field '{price_field}' not found in the DataFrame columns: {df.columns.tolist()}")

        price_series = df[price_field].squeeze() # Predict price based on selected field

        # Return both the target series and the full DataFrame (might be useful)
        return price_series, df

    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        # Add more context to the error
        raise Exception(f"Error loading data from MongoDB for {stock_symbol}: {e}")
    finally:
        if client:
            client.close()


# --- 2. Check for Stationarity using ADF Test ---
def check_stationarity(series):
    """Performs Augmented Dickey-Fuller test to check for stationarity."""
    # Handle potential NaN/inf values before test
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        print("Warning: Series is empty after removing NaN/inf. Cannot check stationarity.")
        return False # Cannot be stationary if empty
    try:
        dftest = adfuller(series, autolag='AIC')
        p_value = dftest[1]
        # print(f"ADF Test p-value: {p_value}") # Optional: for debugging
        if p_value <= 0.05:
            return True # Stationary
        else:
            return False # Non-stationary
    except Exception as e:
        print(f"ADF test failed: {e}")
        return False # Assume non-stationary if test fails

# --- 3. Make Series Stationary (if needed) using Differencing ---
def make_stationary(series, max_diff=2):
    """Makes a time series stationary by differencing up to max_diff times."""
    if check_stationarity(series):
        print("Series is already stationary.")
        return series, 0

    for d in range(1, max_diff + 1):
        print(f"Attempting differencing with d={d}")
        diff_series = series.diff(d).dropna()
        if diff_series.empty:
            print(f"Differencing with d={d} resulted in empty series. Stopping.")
            return None, d-1 # Return None if differencing fails
        if check_stationarity(diff_series):
            print(f"Series became stationary after differencing with d={d}.")
            return diff_series, d

    print(f"Warning: Series did not become stationary after {max_diff} differencing attempts.")
    # Default to d=1 if max_diff attempts fail, but return the corresponding diff_series
    diff_series_d1 = series.diff(1).dropna()
    if diff_series_d1.empty:
        print("Differencing with d=1 also resulted in empty series.")
        return None, 0 # Indicate failure to difference
    else:
        return diff_series_d1, 1


# --- 4. Split Data into Training and Testing Sets ---
def train_test_split(data, test_size=0.2):
    """Splits time series data into training and testing sets."""
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError("Input data must be a pandas Series or DataFrame.")

    if len(data) < 10: # Arbitrary small number, adjust as needed
        raise ValueError(f"Insufficient data for train/test split (only {len(data)} rows).")

    split_index = int(len(data) * (1 - test_size))
    # Ensure split_index makes sense
    if split_index <= 0:
        split_index = 1 # Need at least one training point
    if split_index >= len(data):
        split_index = len(data) - 1 # Need at least one test point

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    # Final check to ensure both sets are non-empty
    if train_data.empty or test_data.empty:
        raise ValueError(f"Train/Test split resulted in empty set(s). Original len: {len(data)}, split_idx: {split_index}, test_size: {test_size}")

    return train_data, test_data


# --- 5. Automatically Select ARIMA (p, d, q) parameters using AIC ---
def get_auto_arima_params(train_series, max_p=3, max_q=3, d=0): # Pass d
    """Automatically finds best ARIMA (p, q) for a given d using AIC minimization."""
    best_aic = float("inf")
    best_order_pq = None # Only find p and q here

    p_range = range(max_p + 1)
    q_range = range(max_q + 1)

    # Use tqdm for progress indication
    pq_combinations = list(product(p_range, q_range))

    print(f"Searching for best ARIMA(p,{d},q) order (max_p={max_p}, max_q={max_q})...")
    for p, q in tqdm(pq_combinations, desc="AIC Search"):
        if p == 0 and q == 0: # Skip (0,d,0) model
             continue
        order = (p, d, q)
        try:
            # Use low_memory=True if memory becomes an issue, may be slower
            model = ARIMA(train_series, order=order)
            # Use a solver that's less likely to fail, e.g., 'lbfgs'
            # Suppress convergence warnings during search
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning) # Also ignore runtime warnings potentially related to convergence
                model_fit = model.fit(method_kwargs={"warn_convergence": False})
            aic = model_fit.aic
            # Check if AIC is valid (sometimes can be NaN or Inf)
            if pd.notna(aic) and aic < best_aic:
                best_aic = aic
                best_order_pq = (p, q)
        except Exception as e: # Catch errors like non-invertibility, etc.
            # print(f"Order {order} failed: {e}") # Optional debug
            continue

    if best_order_pq:
        best_order = (best_order_pq[0], d, best_order_pq[1])
        print(f"Auto search complete. Best Order: ARIMA{best_order} with AIC={best_aic:.2f}")
        return best_order
    else:
        print("Warning: Auto ARIMA parameter search failed. Using default (1, d, 1).")
        return (1, d, 1) # Return default order (p=1, q=1) if search fails


# --- 6. Train ARIMA Model ---
def train_arima_model(train_series, order):
    """Trains an ARIMA model on the training data."""
    print(f"Training ARIMA{order} model...")
    try:
        model = ARIMA(train_series, order=order)
        # Try fitting with default solver first
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                model_fit = model.fit(method_kwargs={"warn_convergence": False})
            except (np.linalg.LinAlgError, ValueError):
                 print("Default solver failed, trying 'lbfgs'...")
                 # Use a different solver if the default fails
                 model_fit = model.fit(method_kwargs={"solver": 'lbfgs', "warn_convergence": False})

        print("ARIMA model training finished.")
        print(model_fit.summary()) # Print model summary
        return model_fit
    except Exception as e:
        print(f"Error training ARIMA model with order {order}: {e}")
        print(traceback.format_exc())
        return None


# --- 7. Evaluate Model on Test Set (Walk-Forward Validation) ---
def evaluate_model(model_fit, train_series, test_series, original_series, diff_order, stock_symbol, order, evaluation_results_collection, start_date, end_date):
    """Evaluates the trained ARIMA model using walk-forward validation and stores results in MongoDB."""
    history = list(train_series)
    test_actual_vals = original_series.iloc[len(train_series):] # Original scale test values
    predictions_diff_scale = [] # Predictions in the differenced scale

    # Store the initial model fit parameters to potentially reuse
    initial_params = model_fit.params
    last_successful_fit = model_fit # Keep track of the last successful fit

    print("Starting Walk-Forward Validation...")
    for t in tqdm(range(len(test_series)), desc="Walk-Forward Validation"):
        # --- FIX ---
        # Use the 'order' tuple passed into the function, which is the best_order found earlier
        current_order = order
        # -----------

        try:
            # Refit the model with the history up to this point
            wf_model = ARIMA(history, order=current_order)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # Use start_params from the last successful fit for faster convergence
                start_params = last_successful_fit.params
                wf_model_fit = wf_model.fit(start_params=start_params, method_kwargs={"warn_convergence": False})

            # Forecast one step ahead
            output = wf_model_fit.forecast(steps=1)
            yhat_diff = output[0]
            predictions_diff_scale.append(yhat_diff)

            # Update the history with the actual observed value (from the differenced test set)
            history.append(test_series.iloc[t])
            # Update the last successful fit object for the next iteration's start_params
            last_successful_fit = wf_model_fit

        except Exception as e:
            print(f"\nError during walk-forward step {t} for order {current_order}: {e}")
            # Handle error: Append NaN or try to forecast using the last successful model without refitting
            try:
                # Option 1: Forecast using the last good model
                print("Attempting forecast with last successful model...")
                last_output = last_successful_fit.forecast(steps=1)
                yhat_diff = last_output[0]
                print(f"Used forecast from last successful model: {yhat_diff}")
            except Exception as forecast_e:
                print(f"Forecast attempt also failed: {forecast_e}. Appending NaN.")
                yhat_diff = np.nan

            predictions_diff_scale.append(yhat_diff)

            # Try to append actual differenced value to history to continue subsequent steps if possible
            try:
                history.append(test_series.iloc[t])
                print("Appended actual value to history despite error.")
            except IndexError:
                print("Error appending to history after failure, stopping walk-forward.")
                break # Stop if we can't even get the next test point


    # Handle NaNs in predictions (e.g., replace with previous value or mean)
    predictions_diff_scale = pd.Series(predictions_diff_scale, index=test_series.index).ffill().bfill().values
    if np.isnan(predictions_diff_scale).any():
         print("Warning: Could not fill all NaNs in predictions. Filling remaining with 0.")
         predictions_diff_scale = np.nan_to_num(predictions_diff_scale)


    # --- Inverse transform predictions ---
    print("Inverse transforming predictions...")
    predictions_original_scale = []
    # Get the necessary previous original values to start the inverse diff
    # Need 'd' values ending at the last training point
    history_orig_for_inverse = list(original_series.iloc[len(train_series)-diff_order : len(train_series)])

    if diff_order > 0:
        if len(history_orig_for_inverse) < diff_order:
            print(f"Error: Not enough historical points ({len(history_orig_for_inverse)}) for inverse diff d={diff_order}")
            # Cannot proceed with inverse transform if history is too short
            return # Or raise an error

        # Ensure actual test values are available
        if len(test_actual_vals) < len(predictions_diff_scale):
            print(f"Warning: Length mismatch between predictions ({len(predictions_diff_scale)}) and actual test values ({len(test_actual_vals)}). Truncating predictions.")
            predictions_diff_scale = predictions_diff_scale[:len(test_actual_vals)]

        for i in range(len(predictions_diff_scale)):
            # Simple inverse difference logic (adjust if using a library function for inverse)
            if diff_order == 1:
                 # value[t] = prediction_diff[t] + value[t-1]
                 yhat_original_scale = predictions_diff_scale[i] + history_orig_for_inverse[-1]
            elif diff_order == 2:
                 # value[t] = prediction_diff[t] + 2*value[t-1] - value[t-2]
                 yhat_original_scale = predictions_diff_scale[i] + 2*history_orig_for_inverse[-1] - history_orig_for_inverse[-2]
            else:
                 # Higher order inverse is complex and often unstable manually.
                 # Using d=1 approx or consider library approach if needed.
                 print(f"Warning: Inverse difference for d={diff_order} is approximated using d=1 logic.")
                 yhat_original_scale = predictions_diff_scale[i] + history_orig_for_inverse[-1]

            predictions_original_scale.append(yhat_original_scale)

            # Append the *actual* test value to the history for the next step's inverse calculation
            # Make sure index i is valid for test_actual_vals
            if i < len(test_actual_vals):
                history_orig_for_inverse.append(test_actual_vals.iloc[i])
            else:
                print(f"Warning: Ran out of actual test values during inverse transform at step {i}.")
                # Handle this? Maybe stop appending or append the prediction itself?
                # Appending the prediction might lead to drift. Let's stop appending actuals.
                # The loop will continue but history_orig_for_inverse won't grow with actuals anymore.


    else: # diff_order == 0
        # If d=0, predictions are already in the original scale
        predictions_original_scale = predictions_diff_scale

    # Convert to numpy array for metric calculations
    predictions_original_scale = np.array(predictions_original_scale)

    # Align actuals and predictions - ensure they have the same length
    min_len_eval = min(len(test_actual_vals), len(predictions_original_scale))
    actuals_original_scale = test_actual_vals.values[:min_len_eval]
    predictions_original_scale = predictions_original_scale[:min_len_eval]

    # --- Calculate Metrics ---
    print("Calculating evaluation metrics...")
    if min_len_eval == 0:
        print("No data points available for evaluation.")
        rmse_val, mae_val, mape_val = np.nan, np.nan, np.nan
        accuracy_val, cm, sensitivity, specificity, precision_val, recall_val, f1_val = np.nan, np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        # Regression Metrics
        rmse_val = np.sqrt(mean_squared_error(actuals_original_scale, predictions_original_scale))
        mae_val = mean_absolute_error(actuals_original_scale, predictions_original_scale)

        # MAPE calculation with check for zero values in actuals
        mask = actuals_original_scale != 0
        valid_actuals = actuals_original_scale[mask]
        valid_predictions = predictions_original_scale[mask]
        if len(valid_actuals) > 0:
             mape_val = np.mean(np.abs((valid_actuals - valid_predictions) / valid_actuals)) * 100
        else:
             mape_val = np.nan # Handle case where all actuals are zero or non-positive

        print(f'\nModel Evaluation (Original Scale - Regression Metrics):')
        print(f'  RMSE: {rmse_val:.4f}')
        print(f'  MAE:  {mae_val:.4f}')
        print(f'  MAPE: {mape_val:.2f}%')

        # Classification Metrics (Predicting Direction: Up/Down)
        if min_len_eval > 1: # Need at least 2 points to calculate difference
            actual_directions = (np.diff(actuals_original_scale) > 0).astype(int)
            predicted_directions = (np.diff(predictions_original_scale) > 0).astype(int)

            # Ensure lengths match after diff
            len_directions = len(actual_directions) # Both have same len now

            if len_directions > 0:
                accuracy_val = accuracy_score(actual_directions, predicted_directions)
                cm = confusion_matrix(actual_directions, predicted_directions, labels=[0, 1]) # Ensure labels are 0 and 1
                precision_val = precision_score(actual_directions, predicted_directions, zero_division=0)
                recall_val = recall_score(actual_directions, predicted_directions, zero_division=0) # Same as sensitivity
                f1_val = f1_score(actual_directions, predicted_directions, zero_division=0)

                # Handle potential division by zero in sensitivity/specificity manually if cm has zeros
                # Ensure cm has shape (2, 2) even if one class is missing
                if cm.shape == (1, 1): # Only one class present
                    if actual_directions[0] == 1: # Only positives
                        cm = np.array([[0, 0], [0, cm[0,0]]]) # [[TN, FP], [FN, TP]]
                    else: # Only negatives
                        cm = np.array([[cm[0,0], 0], [0, 0]])
                elif cm.shape != (2,2): # Unexpected shape, reset
                     cm = np.array([[0, 0], [0, 0]])


                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                print(f'\nModel Evaluation (Direction Prediction - Classification Metrics):')
                print(f"  Confusion Matrix (Rows: Actual, Cols: Predicted):\n{cm}")
                print(f'  Accuracy:    {accuracy_val:.4f}')
                print(f'  Sensitivity: {sensitivity:.4f} (Recall/TPR)')
                print(f'  Specificity: {specificity:.4f} (TNR)')
                print(f'  Precision:   {precision_val:.4f}')
                print(f'  F1-Score:    {f1_val:.4f}')
            else:
                print("\nCannot calculate classification metrics: No direction changes.")
                accuracy_val, cm, sensitivity, specificity, precision_val, recall_val, f1_val = np.nan, np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan
        else:
             print("\nNot enough data points (<=1) after alignment to calculate classification metrics.")
             accuracy_val, cm, sensitivity, specificity, precision_val, recall_val, f1_val = np.nan, np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan


    # --- Feature Importance (Approximation for ARIMA) ---
    print("\nAnalyzing ARIMA components...")
    feature_importance_dict = {}
    model_to_analyze = last_successful_fit # Analyze the parameters of the final model from walk-forward
    try:
        # Get model parameters (coefficients)
        model_params = model_to_analyze.params
        ar_coeffs = model_params[[p for p in model_params.index if p.startswith('ar.L')]]
        ma_coeffs = model_params[[p for p in model_params.index if p.startswith('ma.L')]]

        feature_importance_dict["AR_Order"] = float(order[0])
        feature_importance_dict["Diff_Order"] = float(order[1])
        feature_importance_dict["MA_Order"] = float(order[2])

        # Store coefficients as importance measure
        for param, value in ar_coeffs.items():
            feature_importance_dict[f"coeff_{param}"] = float(value)
        for param, value in ma_coeffs.items():
            feature_importance_dict[f"coeff_{param}"] = float(value)
        if 'sigma2' in model_params:
             feature_importance_dict["variance_sigma2"] = float(model_params['sigma2'])

        # Print summary of components
        print("ARIMA Component Summary (from final walk-forward step):")
        print(f"  AR Order (p): {order[0]}, MA Order (q): {order[2]}, Differencing (d): {order[1]}")
        if not ar_coeffs.empty: print(f"  AR Coefficients:\n{ar_coeffs}")
        if not ma_coeffs.empty: print(f"  MA Coefficients:\n{ma_coeffs}")
        if 'sigma2' in model_params: print(f"  Estimated Variance (sigma2): {model_params['sigma2']:.4f}")

    except Exception as e:
        print(f"Error extracting ARIMA component info: {e}")
        feature_importance_dict = {
            "AR_Order": float(order[0]),
            "Diff_Order": float(order[1]),
            "MA_Order": float(order[2]),
            "error": "Could not extract coefficients"
        }

    # --- Store Trained Model ---
    # Use relative path instead of absolute path
    MODEL_SAVE_DIR_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train-model')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(MODEL_SAVE_DIR_BASE, stock_symbol, f"arima-{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")

    try:
        with open(model_path, 'wb') as file:
            # Save the *final* fitted model from the walk-forward process
            pickle.dump(last_successful_fit, file)
        print(f"\nTrained ARIMA model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")
        model_path = None # Indicate that saving failed

    # --- Store Evaluation Results in MongoDB ---
    evaluation_data = {
        "stock_symbol": stock_symbol,
        "model_type": "ARIMA",
        "trained_timestamp": datetime.now().isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        # Use endog_names if available, otherwise use the input price_field
        "price_field": getattr(last_successful_fit, 'endog_names', args.price_field),
        "feature_engineering_used": not args.skip_feature_engineering,
        "arima_order_p": order[0],
        "arima_order_d": order[1],
        "arima_order_q": order[2],
        # Use np.nan_to_num to store 0 instead of NaN, or handle None for MongoDB
        "rmse": float(np.nan_to_num(rmse_val)) if pd.notna(rmse_val) else None,
        "mae": float(np.nan_to_num(mae_val)) if pd.notna(mae_val) else None,
        "mape": float(np.nan_to_num(mape_val)) if pd.notna(mape_val) else None,
        "accuracy_direction": float(np.nan_to_num(accuracy_val)) if pd.notna(accuracy_val) else None,
        "sensitivity_direction": float(np.nan_to_num(sensitivity)) if pd.notna(sensitivity) else None,
        "specificity_direction": float(np.nan_to_num(specificity)) if pd.notna(specificity) else None,
        "precision_direction": float(np.nan_to_num(precision_val)) if pd.notna(precision_val) else None,
        "f1_score_direction": float(np.nan_to_num(f1_val)) if pd.notna(f1_val) else None,
        "confusion_matrix_direction": cm.tolist() if cm.size > 0 else [],
        "component_importance": feature_importance_dict,
        "model_aic": float(last_successful_fit.aic) if hasattr(last_successful_fit, 'aic') and pd.notna(last_successful_fit.aic) else None,
        "model_bic": float(last_successful_fit.bic) if hasattr(last_successful_fit, 'bic') and pd.notna(last_successful_fit.bic) else None,
        "model_log_likelihood": float(last_successful_fit.llf) if hasattr(last_successful_fit, 'llf') and pd.notna(last_successful_fit.llf) else None,
        "model_save_path": model_path # Store path if saving succeeded
    }
    try:
        # Ensure collection exists or is created
        if evaluation_collection_name not in evaluation_results_collection.database.list_collection_names():
            print(f"Creating MongoDB collection: {evaluation_collection_name}")
            # MongoDB usually creates collection on first insert, but this ensures clarity
            # evaluation_results_collection.database.create_collection(evaluation_collection_name)

        evaluation_results_collection.insert_one(evaluation_data)
        print(f"Evaluation results for {stock_symbol} stored in MongoDB collection '{evaluation_collection_name}'.")
    except Exception as e:
        print(f"Error storing evaluation results in MongoDB: {e}")


# --- Main Program ---
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # --- MongoDB Connection Details (Match XGBoost) ---
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGO_DB_NAME', 'stock_prices') # USE SAME DB AS XGBOOST
    evaluation_collection_name = "arima_evaluation_results" # Separate collection for ARIMA results

    # --- Model Parameters ---
    # Use the default 'Close' (capital C) from args now
    price_field_to_use = args.price_field
    feature_engineering = not args.skip_feature_engineering
    max_p = args.max_p
    max_q = args.max_q
    test_size = args.test_size

    print(f"ARIMA parameters: price_field='{price_field_to_use}', feature_engineering={feature_engineering}")
    print(f"max_p={max_p}, max_q={max_q}, test_size={test_size}")
    if args.auto_diff:
        print("Differencing order (d): Automatic")
    else:
        print(f"Differencing order (d): {args.diff_order} (fixed)")

    # --- Date Range ---
    start_date_str = args.start_date
    end_date_str = args.end_date
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Please use YYYY-MM-DD format for dates.")
        sys.exit(1)

    print(f"Training with date range: {start_date} to {end_date}")

    # --- Load Stock Symbols ---
    stock_symbols = get_stock_symbols(args)
    print(f"Training ARIMA model for {len(stock_symbols)} stock(s): {', '.join(stock_symbols) if len(stock_symbols) < 5 else ', '.join(stock_symbols[:5]) + '...'}")

    client = None
    try:
        # Establish MongoDB connection once
        client = MongoClient(mongo_uri)
        # Test connection
        client.admin.command('ping')
        db = client[db_name]
        evaluation_results_collection = db[evaluation_collection_name]
        print(f"Connected to MongoDB: {mongo_uri}, Database: '{db_name}'")

        for stock_symbol_to_predict in stock_symbols:
            print(f"\n--- Processing Stock: {stock_symbol_to_predict} ---")

            try:
                # 1. Load Data using the adapted function
                stock_series, _ = load_data_from_mongodb(
                    mongo_uri, db_name, stock_symbol_to_predict,
                    price_field=price_field_to_use, # Use the arg value
                    start_date=start_date, end_date=end_date,
                    feature_engineering=feature_engineering
                )
                # We only need the price series for standard ARIMA

                print(f"Loaded {len(stock_series)} data points for {stock_symbol_to_predict}.")
                if stock_series.isnull().any():
                     print("Warning: NaNs detected in loaded series, dropping them.")
                     stock_series.dropna(inplace=True)

                if len(stock_series) < 20: # Need sufficient data for differencing/training/splitting
                     raise ValueError(f"Insufficient data points ({len(stock_series)}) after loading/cleaning.")


                # 2. Handle Stationarity
                original_series = stock_series.copy() # Keep original for evaluation inverse transform
                if args.auto_diff:
                    stationary_series, diff_order = make_stationary(stock_series, max_diff=2)
                    if stationary_series is None:
                        print("Failed to make series stationary automatically. Skipping stock.")
                        continue
                else:
                    # Use fixed differencing order
                    diff_order = args.diff_order
                    if diff_order < 0:
                        raise ValueError("Differencing order cannot be negative.")
                    elif diff_order > 0:
                         stationary_series = stock_series.diff(diff_order).dropna()
                         print(f"Applied fixed differencing d={diff_order}.")
                    else: # diff_order == 0
                         stationary_series = stock_series
                         print("Using original series (d=0). Checking stationarity...")
                         if not check_stationarity(stationary_series):
                              print("Warning: Series is non-stationary but d=0 was specified. ARIMA may struggle.")
                         else:
                              print("Series is stationary (or ADF test passed).")


                if stationary_series.empty:
                     print("Series became empty after differencing. Skipping stock.")
                     continue
                if len(stationary_series) < max(max_p, max_q) + 5: # Need enough points for ARIMA estimation
                    raise ValueError(f"Insufficient data points ({len(stationary_series)}) after differencing for ARIMA estimation.")


                # 3. Split Data (on the potentially differenced series)
                train_data, test_data = train_test_split(stationary_series, test_size=test_size)
                print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")

                # 4. Automatically Set ARIMA Parameters (p, q) for the determined d
                best_order = get_auto_arima_params(train_data, max_p=max_p, max_q=max_q, d=diff_order)

                # 5. Train Initial ARIMA Model (before walk-forward)
                initial_arima_model_fit = train_arima_model(train_data, best_order)
                if initial_arima_model_fit is None:
                    print("Initial ARIMA model training failed. Skipping evaluation and saving for this stock.")
                    continue # Skip to the next stock symbol

                # 6. Evaluate Model using Walk-Forward Validation
                # Pass the *initial* fitted model to start the walk-forward process
                evaluate_model(initial_arima_model_fit, train_data, test_data, original_series, diff_order, stock_symbol_to_predict, best_order, evaluation_results_collection, start_date, end_date)

            except ValueError as ve:
                print(f"Skipping {stock_symbol_to_predict} due to ValueError: {ve}")
                continue
            except Exception as e:
                print(f"Skipping {stock_symbol_to_predict} due to an unexpected error: {e}")
                print(traceback.format_exc()) # Print full traceback for debugging
                continue

    except pymongo.errors.ConnectionFailure as e:
        print(f"CRITICAL: Could not connect to MongoDB at {mongo_uri}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main program loop: {e}")
        print(traceback.format_exc())
    finally:
        if client:
            client.close()
            print("\nMongoDB connection closed.")

    print("\n--- ARIMA Stock Price Prediction and Evaluation Completed for all symbols. ---")