Okay, here is a detailed explanation of each machine learning model used in the project, covering their working principles, algorithm/steps, parameters, mathematical formulations, evaluation metrics, and feature importance analysis, drawing from the provided report and code snippets.

---

## Machine Learning Models Used for Stock Market Prediction

This project employs four distinct machine learning algorithms for predicting stock market price movement direction: ARIMA, LSTM, SVM, and XGBoost. Each model is chosen for its specific strengths in handling time-series data, capturing sequential dependencies, modeling non-linear relationships, and incorporating exogenous factors like sentiment. The models are trained on historical stock data and sentiment analysis derived from financial news articles, leveraging various technical indicators as features.

### 1. ARIMA (Autoregressive Integrated Moving Average) / SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors)

*   **Introduction & Working Principles:**
    *   ARIMA is a statistical model for time series forecasting. It captures temporal dependencies in the data.
    *   **AR (Autoregressive):** Uses the relationship between a time series observation and a number of lagged observations (past values).
    *   **I (Integrated):** Uses differencing to make the time series stationary (removing trends or seasonality).
    *   **MA (Moving Average):** Uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
    *   **SARIMAX:** Extends ARIMA by adding seasonal components (S) and the ability to include exogenous variables (X). In this project's context, sentiment scores from financial news are used as exogenous regressors (Report Page 12, 22).

*   **Algorithm/Steps (as implemented in the project):**
    1.  **Load Data:** Historical stock price data is loaded, focusing on the target price field ('Close' in the code) and other relevant columns like 'Open', 'High', 'Low', 'Volume' (Report Page 30, code `load_data_from_mongodb`).
    2.  **Feature Engineering (Optional):** While the core ARIMA model primarily uses the time series itself, SARIMAX incorporates exogenous variables. The project mentions calculating technical indicators (MA, RSI, MACD, ATR, Price Range, Volume Change) and including sentiment scores (Report Page 22, 30, code `load_data_from_mongodb`). However, the standard ARIMA model implemented in `train-arima.py` only uses the differenced *price series* itself and sentiment as an *exogenous* variable, not other technical indicators as direct inputs to the time series component. The evaluation section (Report Page 34) notes that sentiment scores are used as input features, implying they are the exogenous variables for SARIMAX.
    3.  **Check/Make Stationary:** The time series data (price) is checked for stationarity using the Augmented Dickey-Fuller (ADF) test. If non-stationary, differencing is applied (`d` order) to remove trends and seasonality (Report Page 22, 30, code `check_stationarity`, `make_stationary`). The `d` order can be automatically determined or fixed.
    4.  **Split Data:** The stationary or original time series is split into training and testing sets chronologically (e.g., 80% train, 20% test in report Page 22, 31, though code uses 80/20 train/test in `train_test_split`).
    5.  **Auto Parameter Selection:** The optimal non-seasonal `(p, q)` orders are automatically selected for the determined `d` order by minimizing the Akaike Information Criterion (AIC) on the training data (Report Page 22, 32, code `get_auto_arima_params`). The search covers ranges for `p` and `q` (default `max_p=3`, `max_q=3` in code).
    6.  **Train Model:** An ARIMA(`p`, `d`, `q`) model is trained on the differenced (or original) training series using the selected order. The project mentions SARIMAX, implying sentiment scores are used here as exogenous variables during training (`statsmodels.tsa.arima.model.ARIMA` supports `exog` parameter, though not explicitly shown in the provided `train_arima_model` snippet).
    7.  **Evaluate:** The model is evaluated on the test set using a **walk-forward validation** approach. The model is iteratively refitted on the growing history of the training data plus previously observed test values to predict the next step (Report Page 22, code `evaluate_model`).
    8.  **Inverse Transform:** Predictions, which are in the differenced scale, are inverse transformed back to the original price scale for meaningful interpretation and evaluation (Report Page 22, code `evaluate_model`).
    9.  **Calculate Metrics:** Regression metrics (RMSE, MAE, MAPE, R2) are calculated on the original scale predictions. Classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Sensitivity, Specificity) are calculated based on the predicted vs. actual *direction* of price movement (Up/Down), derived from the original scale values (Report Page 23, 31, 32, code `evaluate_model`).
    10. **Analyze Feature Importance:** Component parameters (like AR and MA coefficients, order) are analyzed as a form of interpretability (Report Page 34).
    11. **Save Model & Results:** The trained model (specifically the final fitted model from walk-forward in the code) and evaluation results are saved (Report Page 23, code `evaluate_model`, `save_model` although saving logic is inside `evaluate_model`).

*   **Parameters Used (based on project report and code defaults/arguments):**
    *   `(p, d, q)` orders: Automatically selected (`p` up to 3, `q` up to 3) based on AIC minimization for a determined differencing order `d` (0 or 1, determined by ADF test or fixed by argument) (Report Page 22, 32, code `get_auto_arima_params`). The report mentions SARIMAX with seasonality `(P, D, Q)s` and `s=365`, but the core `train-arima.py` script implements a non-seasonal ARIMA with exogenous variables.
    *   Exogenous Variables: Daily Sentiment Score (from FinBERT-tone) (Report Page 22).
    *   Evaluation Test Size: 20% of data (Report Page 22).
    *   Maximum Differencing Attempts: 2 (code `make_stationary`).

*   **Mathematical Formulation:**
    *   A standard ARIMA(p,d,q) model can be written as:
        $(1 - \sum_{i=1}^p \phi_i B^i) (1-B)^d y_t = (1 + \sum_{j=1}^q \theta_j B^j) \epsilon_t$
        where:
        *   $y_t$ is the time series at time $t$.
        *   $B$ is the backshift operator ($By_t = y_{t-1}$).
        *   $\phi_i$ are the autoregressive coefficients.
        *   $\theta_j$ are the moving average coefficients.
        *   $\epsilon_t$ is the error term (white noise).
        *   $d$ is the differencing order.
    *   SARIMAX extends this by adding seasonal components and exogenous regressors. The specific example formulation shown in the report (Page 23) is:
        $(1 − \phi_1 B) (1 – \Phi_1 B^4) (1 − B) (1 – B^4)y_t = (1 + \theta_1 B) (1 + \Theta_1 B^4)\epsilon_t$
        This represents a SARIMA(1,1,1)(1,1,1) with seasonality 4 (which might be quarterly or weekly data, not daily 365 as mentioned elsewhere in the report, suggesting a possible inconsistency in the example). The actual implementation likely uses SARIMAX(p,d,q) with exogenous variables (sentiment) but *without* seasonal differencing/AR/MA components as per the `get_auto_arima_params` code structure.

*   **Evaluation Metrics:** RMSE, MAE, MAPE, R2 (Regression), Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity, Confusion Matrix (Classification) (Report Page 23, 31, 32).

*   **Feature Importance:** Analysis of the fitted model's coefficients and estimated variance (`sigma2`) as indicators of the influence of past values and errors (Report Page 34, code `evaluate_model`).

### 2. LSTM (Long Short-Term Memory) Networks

*   **Introduction & Working Principles:**
    *   LSTMs are a type of Recurrent Neural Network (RNN) designed specifically for sequential data like time series.
    *   They are capable of learning long-term dependencies, overcoming the vanishing gradient problem common in standard RNNs.
    *   LSTMs use a complex internal mechanism called "gates" (Forget, Input, Output) and a "memory cell" to regulate the flow of information, deciding what data to keep, what to discard, and what to pass on (Report Page 9, 20).

*   **Algorithm/Steps (as implemented in the project):**
    1.  **Load Data:** Historical stock price data (Open, High, Low, Close, Volume) is loaded from MongoDB (Report Page 12, 19, 27, 30, code `load_data_from_mongodb`).
    2.  **Feature Engineering:** Technical indicators (SMA, EMA, Lagged Returns - Report Page 21, code `calculate_technical_indicators`) are calculated. Sentiment scores (derived from FinBERT-tone analysis of news) are also integrated as input features (Report Page 17, 21, 30).
    3.  **Feature Selection:** Relevant features are selected (code `train_lstm_model` explicitly selects 'Close', 'SMA_Short', 'SMA_Long', 'EMA_Short', 'EMA_Long', 'Lagged_Return_1').
    4.  **Scaling:** Numerical features are scaled (e.g., using MinMaxScaler) to a standard range [0, 1] to improve model performance and prevent features with large values from dominating (Report Page 21, 30, code `MinMaxScaler`).
    5.  **Create Sequences:** The scaled feature data is transformed into sequences of a fixed length (`seq_length`) suitable for LSTM input. Each sequence corresponds to a window of historical data, and the target is the value at the end of or just after the sequence (Report Page 21, 32, code `create_sequences`).
    6.  **Define Target Labels:** For classification evaluation, binary labels (Up/Down) are created based on whether the target price is higher or lower than a lagged price (e.g., the previous day's close) (Report Page 21, code `create_classification_labels`). The regression model predicts the next price itself.
    7.  **Split Data:** Data sequences are split into training and testing sets chronologically (e.g., 80% train, 20% test) (Report Page 21, 31, code `train_test_split`).
    8.  **Build Model:** A sequential Keras model is built, typically with stacked LSTM layers (2 stacked layers in code), activation functions (ReLU), Dropout layers for regularization, and a final Dense output layer (1 unit for regression) (Report Page 21, code `model_reg = Sequential(...)`).
    9.  **Compile Model:** The model is compiled with an optimizer (Adam) and a loss function suitable for regression (Mean Squared Error - MSE) (Report Page 21, code `model_reg.compile(...)`).
    10. **Train Model:** The model is trained on the training sequences for a specified number of epochs and batch size. Callbacks like Early Stopping and Model Checkpointing are used to prevent overfitting and save the best model based on validation loss (Report Page 21, code `model_reg.fit(...)`).
    11. **Evaluate:** The trained model predicts on the test sequences.
    12. **Inverse Transform:** Regression predictions are inverse transformed back to the original price scale using the fitted scaler (Report Page 24, code `evaluate_regression` within `train_lstm_model`).
    13. **Calculate Metrics:** Regression metrics (MSE, RMSE, MAE, MAPE, R2) are calculated on original scale predictions. Classification metrics (Accuracy, Confusion Matrix, Classification Report, Sensitivity, Specificity, Precision, F1-Score) are calculated by comparing the derived Up/Down predictions (based on regression output vs lagged price) against the true Up/Down labels (Report Page 21, 31, 32, code `evaluate_regression`, `evaluate_classification`).
    14. **Analyze Feature Importance:** SHAP (SHapley Additive exPlanations) values are calculated using `shap.DeepExplainer` on a subset of the test data to understand the contribution of each input feature (across the time sequence) to the model's predictions (Report Page 34, code `generate_lstm_shap_values`).
    15. **Save Model & Scalers:** The trained model weights (`.h5` file), the scaler object, and the list of selected features are saved for future inference (Report Page 23, code `ModelCheckpoint`, `save_model` within `train_lstm_model`).

*   **Parameters Used (based on project report and code defaults/arguments):**
    *   `seq_length`: Length of input sequences (default 60) (Report Page 21, code argument `seq-length`).
    *   `lstm_units`: Number of units in LSTM layers (default 60) (Report Page 21, code argument `lstm-units`).
    *   `dropout_rate`: Dropout rate (default 0.2) (Report Page 21, code argument `dropout-rate`).
    *   `epochs`: Number of training epochs (default 75) (Report Page 21, code argument `epochs`).
    *   `batch_size`: Batch size (default 32) (Report Page 21, code argument `batch-size`).
    *   `patience`: Patience for Early Stopping (default 15) (code argument `patience`).
    *   Feature Windows: `short_window` (default 20), `long_window` (default 50) for MAs/EMAs (code arguments).
    *   `lag_days`: Number of days for lagged features (default 1) (code argument `lag-days`).
    *   Test Size: 20% (chronological split) (Report Page 21).

*   **Mathematical Formulation (LSTM Cell):**
    *   Forget Gate: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
    *   Input Gate: $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
    *   Candidate Cell State: $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$
    *   Cell State Update: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
    *   Output Gate: $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
    *   Hidden State: $h_t = o_t * \tanh(C_t)$
    *   Where $\sigma$ is the sigmoid function, $W$ are weights, $b$ are biases, $h_{t-1}$ is the previous hidden state, $x_t$ is the current input, $C_{t-1}$ is the previous cell state, and $*$ denotes element-wise multiplication. The final prediction is typically derived from the last hidden state or a Dense layer connected to it.

*   **Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity, Confusion Matrix (Classification) (Report Page 21, 31, 32, code `evaluate_regression`, `evaluate_classification`).

*   **Feature Importance:** SHAP values calculated using `DeepExplainer` for LSTM models (Report Page 34, code `generate_lstm_shap_values`).

### 3. SVM (Support Vector Machine) for Regression (SVR)

*   **Introduction & Working Principles:**
    *   SVM is a supervised learning model used for both classification and regression.
    *   **SVR:** An adaptation for regression tasks. Instead of finding a hyperplane that separates classes, SVR finds a hyperplane that best fits the data points while maximizing the margin between data points and the hyperplane within a specified error tolerance (epsilon, $\epsilon$) (Report Page 9, 12, 23).
    *   **Kernel Trick:** SVR uses kernel functions (like RBF, linear, polynomial) to implicitly map the input data into a higher-dimensional feature space, allowing it to capture non-linear relationships (Report Page 12, 23).

*   **Algorithm/Steps (as implemented in the project):**
    1.  **Load Data:** Historical stock price data (Open, High, Low, Close, Volume) is loaded from MongoDB (Report Page 12, 19, 27, 30, code `load_data_from_mongodb`).
    2.  **Prepare Data:** Features are engineered, including technical indicators (SMA, EMA, Price Range, Volume Change, Return) and lagged features (Close prices from previous `look_back` days). Sentiment scores are also included as features (Report Page 24, code `prepare_data_for_svm`).
    3.  **Define Target:** The target variable (`y`) is the closing price of the *next* day, created by shifting the 'Close' column upwards (code `prepare_data_for_svm`).
    4.  **Separate X and y:** Features are separated into `X` and the target into `y` (code `prepare_data_for_svm`).
    5.  **Scaling:** Features (`X`) and the target (`y`) are scaled using StandardScaler. Separate scalers are used for close/lagged features, other features, and the target (Report Page 24, code `prepare_data_for_svm`).
    6.  **Feature Selection:** A subset of the most relevant features is selected using a method like SelectKBest based on regression correlation (`f_regression`) (code `prepare_data_for_svm`).
    7.  **Split Data:** The data is split into training and testing sets chronologically (e.g., 80% train, 20% test) (Report Page 24, 31, code `train_test_split` within `prepare_data_for_svm`).
    8.  **Train Model:** An SVR model is trained on the scaled training data. Hyperparameter tuning (GridSearchCV) is performed to find the best combination of parameters like `C`, `gamma`, `epsilon`, and `kernel` by minimizing Mean Squared Error (MSE) on a validation set or cross-validation (Report Page 24, 32, 33, code `train_svm_model`).
    9.  **Evaluate:** The trained model predicts on the scaled test features.
    10. **Inverse Transform:** Predictions (`y_pred`) and actual test values (`y_test`) are inverse transformed back to their original price scale using the fitted target scaler (Report Page 24, code `evaluate_model`).
    11. **Calculate Metrics:** Regression metrics (MSE, RMSE, MAE, MAPE, R2) are calculated on the original scale values. Classification metrics (Accuracy, Confusion Matrix, Classification Report) are calculated based on the predicted vs. actual *direction* of price movement (Up/Down), derived from the original scale values (Report Page 24, 31, 32, code `evaluate_model`).
    12. **Analyze Feature Importance:** SHAP (SHapley Additive exPlanations) values are calculated using `shap.KernelExplainer` on a subset of the test data to understand the contribution of each input feature to the model's predictions (Report Page 34, code `evaluate_model`).
    13. **Save Model & Scalers:** The trained SVR model, the feature scalers (close, other, target), and the list of selected features are saved using pickle (Report Page 24, code `save_model`).

*   **Parameters Used (based on project report and code defaults/arguments):**
    *   `look_back`: Look-back period for features (default 60) (code argument `look-back`).
    *   `num_features_to_select`: Number of features to select (default 30) (code argument `features`).
    *   `kernel`: Kernel type ('linear', 'poly', 'rbf', 'sigmoid') (default 'rbf') (Report Page 33, code argument `kernel`).
    *   `C`: Regularization parameter (grid search over default values like 0.1, 1, 10, 100) (Report Page 24, 33, code argument `c-values`).
    *   `gamma`: Kernel coefficient (grid search over default values like 'scale', 'auto', 0.1, 1, 10) (Report Page 24, 33, code argument `gamma-values`).
    *   `epsilon`: Margin of tolerance (grid search over default values like 0.01, 0.1, 1) (Report Page 24, 33).
    *   Test Size: 20% (chronological split) (Report Page 24).

*   **Mathematical Formulation (SVR Objective):**
    SVR aims to find a function $f(x) = w \cdot \phi(x) + b$ that minimizes the objective function:
    $\min_{w, b, \zeta, \zeta^*} \frac{1}{2} ||w||^2 + C \sum_{i=1}^m (\zeta_i + \zeta_i^*)$
    subject to:
    $y_i - f(x_i) \le \epsilon + \zeta_i$
    $f(x_i) - y_i \le \epsilon + \zeta_i^*$
    $\zeta_i, \zeta_i^* \ge 0$
    *   Where $w$ is the weight vector, $b$ is the bias, $\phi(x)$ is the mapped feature vector in the higher dimension, $\epsilon$ is the margin of tolerance, $C$ is the regularization parameter, $\zeta_i, \zeta_i^*$ are slack variables (penalizing points outside the $\epsilon$-tube), $y_i$ is the actual value, and $f(x_i)$ is the predicted value.

*   **Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Confusion Matrix, Classification Report (Classification) (Report Page 24, 31, 32, code `evaluate_model`). Note: Precision, Recall, F1, Sensitivity, Specificity are part of the Classification Report but not explicitly returned as separate variables in the provided `evaluate_model` snippet for SVM, although they are likely calculated within `classification_report`.

*   **Feature Importance:** SHAP values calculated using `KernelExplainer` for SVM models (Report Page 34, code `evaluate_model`).

### 4. XGBoost (Extreme Gradient Boosting)

*   **Introduction & Working Principles:**
    *   XGBoost is a popular gradient boosting algorithm known for its performance and speed.
    *   It is an ensemble method that builds a sequence of decision trees. Each new tree attempts to correct the errors made by the combination of previous trees.
    *   It uses gradient descent to minimize the loss function and includes regularization terms (L1 and L2) in the objective function to prevent overfitting (Report Page 10, 14, 25).

*   **Algorithm/Steps (as implemented in the project):**
    1.  **Load Data:** Historical stock price data (Open, High, Low, Close, Volume) is loaded from MongoDB (Report Page 12, 19, 27, 30, code `load_data_from_mongodb`).
    2.  **Prepare Data:** Features are engineered, including technical indicators (SMA, EMA, Price Range, Volume Change, Return) and lagged features (Close prices from previous `look_back` days). Sentiment scores are also included as features (Report Page 25, code `prepare_data_for_xgboost`). This is the same feature engineering logic as used for SVM in the code.
    3.  **Define Target:** The target variable (`y`) is the closing price of the *next* day, created by shifting the 'Close' column upwards (code `prepare_data_for_xgboost`).
    4.  **Separate X and y:** Features are separated into `X` and the target into `y` (code `prepare_data_for_xgboost`).
    5.  **Scaling:** Features (`X`) and the target (`y`) are scaled using StandardScaler. Separate scalers are used for close/lagged features, other features, and the target (Report Page 25, code `prepare_data_for_xgboost`).
    6.  **Feature Selection:** A subset of the most relevant features is selected using a method like SelectKBest based on regression correlation (`f_regression`) (Report Page 25, code `prepare_data_for_xgboost`).
    7.  **Split Data:** The data is split into training and testing sets chronologically (e.g., 80% train, 20% test) (Report Page 25, 31, code `train_test_split` within `prepare_data_for_xgboost`).
    8.  **Train Model:** An XGBoost Regressor model is trained on the scaled training data. Hyperparameter tuning (GridSearchCV) is performed to find the best combination of parameters like `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda` by minimizing Mean Squared Error (MSE) on a validation set or cross-validation (Report Page 25, 32, 33, code `train_xgboost_model`).
    9.  **Evaluate:** The trained model predicts on the scaled test features.
    10. **Inverse Transform:** Predictions (`y_pred`) and actual test values (`y_test`) are inverse transformed back to their original price scale using the fitted target scaler (Report Page 26, code `evaluate_model`).
    11. **Calculate Metrics:** Regression metrics (MSE, RMSE, MAE, MAPE, R2) are calculated on the original scale values. Classification metrics (Accuracy, Confusion Matrix, Classification Report) are calculated based on the predicted vs. actual *direction* of price movement (Up/Down), derived from the original scale values (Report Page 25, 31, 32, code `evaluate_model`). Note: Precision, Recall, F1, Sensitivity, Specificity are part of the Classification Report but not explicitly returned as separate variables in the provided `evaluate_model` snippet for XGBoost.
    12. **Analyze Feature Importance:** Both XGBoost's built-in `feature_importances_` attribute and SHAP (SHapley Additive exPlanations) values (using `shap.TreeExplainer`) are used to determine the relative importance of different input features (Report Page 25, 34, code `evaluate_model`).
    13. **Save Model & Scalers:** The trained XGBoost model, the feature scalers (close, other, target), and the list of selected features are saved using pickle (Report Page 25, code `save_model`).

*   **Parameters Used (based on project report and code defaults/arguments):**
    *   `look_back`: Look-back period for features (default 60) (code argument `look-back`).
    *   `num_features_to_select`: Number of features to select (default 30) (code argument `features`).
    *   `n_estimators`: Number of boosting rounds (grid search over default values like 100, 200, 300) (Report Page 25, 32, code argument `n-estimators`).
    *   `max_depth`: Maximum tree depth (grid search over default values like 3, 4, 5) (Report Page 25, 32, code argument `max-depth`).
    *   `learning_rate`: Step size shrinkage (grid search over default values like 0.01, 0.1, 0.2) (Report Page 25, 32, code argument `learning-rate`).
    *   `subsample`: Fraction of training instances used per tree (grid search over values like 0.7, 0.8, 0.9, 1.0) (Report Page 33).
    *   `colsample_bytree`: Fraction of features used per tree (grid search over values like 0.7, 0.8, 0.9, 1.0) (Report Page 33).
    *   `gamma`: Minimum loss reduction for split (grid search over values like 0, 0.1, 0.2) (Report Page 33).
    *   `reg_alpha` (L1) and `reg_lambda` (L2): Regularization terms (grid search over values) (Report Page 33).
    *   Test Size: 20% (chronological split) (Report Page 25).

*   **Mathematical Formulation:**
    *   **Prediction:** The prediction $\hat{y}_i$ for an instance $x_i$ is the sum of the predictions from $K$ trees:
        $\hat{y}_i = \sum_{k=1}^K f_k(x_i)$
        where $f_k$ is the prediction of the $k$-th tree.
    *   **Regularized Objective Function:** XGBoost minimizes a regularized objective function during training, which includes both the loss function (measuring prediction error) and regularization terms (penalizing model complexity):
        $obj(\theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$
        where:
        *   $l(y_i, \hat{y}_i)$ is the loss function (e.g., squared error for regression) measuring the difference between the actual value $y_i$ and the prediction $\hat{y}_i$.
        *   $\Omega(f_k)$ is the regularization term for the $k$-th tree, which penalizes complexity (e.g., number of leaves, magnitude of leaf weights).
        *   $\sum_{i=1}^n$ sums over all training instances.
        *   $\sum_{k=1}^K$ sums over all trees.

*   **Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Confusion Matrix, Classification Report (Classification) (Report Page 25, 31, 32, code `evaluate_model`). Note: As with SVM, classification details like Precision/Recall are part of the report text but not separate variables in the code's `evaluate_model` return for XGBoost.

*   **Feature Importance:** XGBoost's built-in feature importance scores (based on gain, coverage, or frequency) and SHAP values calculated using `TreeExplainer` are used (Report Page 34, code `evaluate_model`).

---

These detailed descriptions align with the project report and code, providing a comprehensive view of how each model is applied to the task of stock market prediction, including the specific data processing, training, and evaluation procedures followed.