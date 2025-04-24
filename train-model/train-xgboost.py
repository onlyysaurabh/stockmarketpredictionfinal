# train-xgboost.py

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV
import pymongo
from pymongo import MongoClient
import pickle
import os
import numpy as np
import sys
import argparse
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_regression

# Add parent directory to path to import mongo_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stocks.mongo_utils import get_mongo_db, STOCKS_COLLECTION


# --- 1. Load Stock Price Data from MongoDB ---
def load_data_from_mongodb(mongo_uri, db_name, stock_symbol,
                           start_date=None, end_date=None):
    """Loads stock data from MongoDB, with optional date range."""
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
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
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

        # Set Date as index
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True) # Ensure data is sorted by date

        return df

    except pymongo.errors.ConnectionFailure as e:
        raise Exception(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        raise Exception(f"Error loading data from MongoDB: {e}")
    finally:
        if client:
            client.close()


# --- 2. Prepare Data for XGBoost ---
def prepare_data_for_xgboost(df, look_back=60, num_features_to_select=30):
    """Prepares data: creates features, scales, selects, and splits."""

    if 'Close' not in df.columns:
        raise ValueError("Required field 'Close' not found in DataFrame.")

    # 1. Feature Engineering (on a *copy*)
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=look_back).mean()
    df['EMA'] = df['Close'].ewm(span=look_back, adjust=False).mean()
    df['Price_Range'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].diff()
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)  # Drop NaNs *after* feature engineering

    # 2. Create Lagged Features
    for i in range(1, look_back + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)

    # 3. Define Target Variable
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    # 4. Separate Features (X) and Target (y)
    X = df.drop('Target', axis=1)
    y = df['Target']

    # 5. Feature Scaling
    close_features = ['Close'] + [f'Close_Lag_{i}' for i in range(1, look_back + 1)]
    other_features = [col for col in X.columns if col not in close_features]

    close_scaler = StandardScaler()
    X.loc[:, close_features] = close_scaler.fit_transform(X[close_features])

    other_scaler = StandardScaler()
    if other_features:  # Handle no "other" features case
        # Explicitly cast to float64 before scaling
        X.loc[:, other_features] = other_scaler.fit_transform(X[other_features].astype(np.float64))

    # 6. Feature Selection
    if num_features_to_select < X.shape[1]:
        selector = SelectKBest(score_func=f_regression,
                               k=num_features_to_select)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_selected, index=X.index,
                         columns=selected_features)  # Keep index
    else:
        selected_features = X.columns.tolist()


    # 7. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 8. Scale Target (for inverse transform)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    return (X_train, X_test, y_train_scaled, y_test_scaled, close_scaler,
            other_scaler if other_features else None, target_scaler, selected_features)


# --- Parse Command Line Arguments ---
def parse_args():
    """Parse command line arguments for training parameters."""
    parser = argparse.ArgumentParser(description='Train XGBoost model for stock price prediction')
    
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
    parser.add_argument('--look-back', type=int, default=60,
                      help='Look-back period for time series features (default: 60)')
    parser.add_argument('--features', type=int, default=30,
                      help='Number of features to select (default: 30)')
    parser.add_argument('--n-estimators', type=str, default="100,200,300",
                      help='Comma-separated list of n_estimators to try (default: 100,200,300)')
    parser.add_argument('--max-depth', type=str, default="3,4,5",
                      help='Comma-separated list of max_depth to try (default: 3,4,5)')
    parser.add_argument('--learning-rate', type=str, default="0.01,0.1,0.2",
                      help='Comma-separated list of learning rates to try (default: 0.01,0.1,0.2)')
                      
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

# --- 3. Train XGBoost Model with Custom Parameters ---
def train_xgboost_model(X_train, y_train, n_estimators=None, max_depth=None, learning_rate=None):
    """Trains XGBoost model with hyperparameter tuning (GridSearchCV)."""
    # Parse parameter lists from strings if provided
    if isinstance(n_estimators, str):
        n_estimators = [int(x) for x in n_estimators.split(',')]
    if isinstance(max_depth, str):
        max_depth = [int(x) for x in max_depth.split(',')]
    if isinstance(learning_rate, str):
        learning_rate = [float(x) for x in learning_rate.split(',')]
        
    param_grid = {
        'n_estimators': n_estimators if n_estimators else [100, 200, 300],
        'max_depth': max_depth if max_depth else [3, 4, 5],
        'learning_rate': learning_rate if learning_rate else [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    model = xgb.XGBRegressor(objective='reg:squarederror',
                             random_state=42,  # For reproducibility
                             n_jobs=-1) # Use all available cores
    grid_search = GridSearchCV(model, param_grid, cv=3,
                               scoring='neg_mean_squared_error', verbose=0,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


# --- 4. Evaluate Model ---
def evaluate_model(model, X_test, y_test, close_scaler, other_scaler, target_scaler, stock_symbol, start_date, end_date, evaluation_results_collection):
    """Evaluates, inverse transforms, and stores results in MongoDB."""
    y_pred = model.predict(X_test)

    # Inverse transform
    y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate regression metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

    print("\nRegression Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R-squared (R2): {r2:.4f}")

    # Classification metrics
    y_test_class = np.where(np.diff(y_test_orig) > 0, 1, 0)
    y_pred_class = np.where(np.diff(y_pred_orig) > 0, 1, 0)

    if len(y_test_orig) > 1 and len(y_pred_orig) > 1:
        min_len = min(len(y_test_class), len(y_pred_class))
        y_test_class = y_test_class[:min_len]
        y_pred_class = y_pred_class[:min_len]

        accuracy = accuracy_score(y_test_class, y_pred_class)
        conf_matrix = confusion_matrix(y_test_class, y_pred_class)
        class_report = classification_report(y_test_class, y_pred_class, zero_division=0)
        print("\nClassification Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print("  Confusion Matrix:\n", conf_matrix)
        print("  Classification Report:\n", class_report)
    else:
        accuracy = 0
        conf_matrix = np.array([])
        class_report = "Not enough data for classification metrics."
        print("\nNot enough data for classification metrics.")

    # Calculate feature importance from XGBoost
    feature_importance = model.feature_importances_
    feature_importance_dict = {}
    for i, importance in enumerate(feature_importance):
        if i < len(X_test.columns):
            feature_importance_dict[X_test.columns[i]] = float(importance)
    
    # Print top 10 features by importance
    print("\nXGBoost Feature Importance (Top 10):")
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in sorted_features:
        print(f"  {feature}: {importance:.4f}")

    # Store evaluation results
    best_params = model.get_params()
    evaluation_data = {
      "stock_symbol": stock_symbol,
      "start_date": start_date.isoformat(),
      "end_date": end_date.isoformat(),
      "xgb_n_estimators": best_params.get('n_estimators', None), # Get hyperparams
      "xgb_max_depth": best_params.get('max_depth', None),
      "xgb_learning_rate": best_params.get('learning_rate', None),
      "xgb_subsample": best_params.get('subsample', None),
      "xgb_colsample_bytree": best_params.get('colsample_bytree', None),
      "mse": mse,
      "rmse": rmse,
      "mae": mae,
      "mape": mape,
      "r2": r2,
      "accuracy": accuracy,
      "confusion_matrix": conf_matrix.tolist() if conf_matrix.size > 0 else [],
      "report": class_report,
      "feature_importance": feature_importance_dict
    }
    try:
        evaluation_results_collection.insert_one(evaluation_data)
        print(f"Evaluation results for {stock_symbol} stored in MongoDB.")
    except Exception as e:
        print(f"Error storing evaluation results in MongoDB: {e}")

    return mse, rmse, mae, r2, mape, accuracy, conf_matrix, class_report


# --- 5. Save Trained Model ---
def save_model(model, stock_symbol, close_scaler, other_scaler, target_scaler, selected_features, model_name="xgboost"):
    """Saves the model, scalers, and selected features."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Use relative path within the project structure
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_dir, "train-model", stock_symbol, f"{model_name}-{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    close_scaler_path = os.path.join(model_dir, "close_scaler.pkl")
    other_scaler_path = os.path.join(model_dir, "other_scaler.pkl")
    target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
    features_path = os.path.join(model_dir, "selected_features.pkl")

    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Trained XGBoost model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model to: {model_path}: {e}")

    try:
        with open(close_scaler_path, 'wb') as file:
            pickle.dump(close_scaler, file)
        print(f"Close scaler saved to: {close_scaler_path}")
    except Exception as e:
        print(f"Error saving close scaler to: {close_scaler_path}: {e}")

    if other_scaler is not None:
        try:
            with open(other_scaler_path, 'wb') as file:
                pickle.dump(other_scaler, file)
            print(f"Other features scaler saved to: {other_scaler_path}")
        except Exception as e:
            print(f"Error saving other features scaler: {other_scaler_path}: {e}")

    try:
        with open(target_scaler_path, 'wb') as file:
            pickle.dump(target_scaler, file)
        print(f"Target scaler saved to: {target_scaler_path}")
    except Exception as e:
        print(f"Error saving target scaler to: {target_scaler_path}: {e}")

    try:
        with open(features_path, 'wb') as file:
            pickle.dump(selected_features, file)
        print(f"Selected features saved to: {features_path}")
    except Exception as e:
        print(f"Error saving selected features to: {features_path}: {e}")


# --- Main Program ---
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # --- MongoDB Connection Details ---
    # Fix the database name to match what's used in the services.py file
    import os
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGO_DB_NAME', 'stock_prices')
    evaluation_collection_name = "xgboost_evaluation_results"

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
    print(f"Training XGBoost model for {len(stock_symbols)} stock(s): {', '.join(stock_symbols) if len(stock_symbols) < 5 else ', '.join(stock_symbols[:5]) + '...'}")

    # --- Model Parameters ---
    look_back = args.look_back
    num_features = args.features
    print(f"Using look_back={look_back}, num_features={num_features}")
    print(f"Hyperparameter tuning with: n_estimators={args.n_estimators}, max_depth={args.max_depth}, learning_rate={args.learning_rate}")

    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        evaluation_results_collection = db[evaluation_collection_name]

        for stock_symbol_to_predict in stock_symbols:
            print(f"\n--- Processing Stock: {stock_symbol_to_predict} ---")
            try:
                # 1. Load Data
                df = load_data_from_mongodb(
                    mongo_uri, db_name,
                    stock_symbol_to_predict, start_date, end_date
                )

                # 2. Prepare Data
                (X_train, X_test, y_train_scaled, y_test_scaled,
                 close_scaler, other_scaler, target_scaler, selected_features) = prepare_data_for_xgboost(
                    df, look_back=look_back, num_features_to_select=num_features
                )

                # 3. Train Model with custom parameters
                xgboost_model = train_xgboost_model(X_train, y_train_scaled, 
                                                  args.n_estimators, args.max_depth, args.learning_rate)

                # 4. Evaluate and Store Results
                evaluate_model(xgboost_model, X_test, y_test_scaled,
                               close_scaler, other_scaler, target_scaler,
                               stock_symbol_to_predict, start_date, end_date,
                               evaluation_results_collection)

                # 5. Save Model and Scalers
                save_model(xgboost_model, stock_symbol_to_predict,
                           close_scaler, other_scaler, target_scaler, selected_features)

            except ValueError as ve:
                print(f"Value Error for {stock_symbol_to_predict}: {ve}. "
                      f"Skipping stock.")
                continue
            except Exception as e:
                print(f"An error occurred for {stock_symbol_to_predict}: {e}. "
                      f"Skipping stock.")
                continue

    except pymongo.errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in main program: {e}")
    finally:
        if client:
            client.close()

    print("\n--- XGBoost Stock Price Prediction and Evaluation Completed. ---")
