# Machine Learning Models Used for Stock Market Prediction

This project employs four distinct machine learning algorithms for predicting stock market price movement direction: **ARIMA**, **LSTM**, **SVM**, and **XGBoost**. Each model is chosen for its specific strengths in handling time-series data, capturing sequential dependencies, modeling non-linear relationships, and incorporating exogenous factors like sentiment. The models are trained on historical stock data and sentiment analysis derived from financial news articles, leveraging various technical indicators as features.

---

## 1. ARIMA (Autoregressive Integrated Moving Average) / SARIMAX (Seasonal ARIMA with Exogenous Regressors)

**Introduction & Working Principles:**
- ARIMA is a statistical model for time series forecasting. It captures temporal dependencies in the data.
- **AR (Autoregressive):** Uses the relationship between a time series observation and a number of lagged observations (past values).
- **I (Integrated):** Uses differencing to make the time series stationary (removing trends or seasonality).
- **MA (Moving Average):** Uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
- **SARIMAX:** Extends ARIMA by adding seasonal components (S) and the ability to include exogenous variables (X). In this project's context, sentiment scores from financial news are used as exogenous regressors.

**Algorithm/Steps:**
1. Load Data: Historical stock price data is loaded, focusing on the target price field ('Close') and other relevant columns.
2. Feature Engineering (Optional): SARIMAX incorporates exogenous variables (sentiment scores). Technical indicators may be calculated.
3. Check/Make Stationary: The time series data is checked for stationarity using the Augmented Dickey-Fuller (ADF) test. Differencing is applied if needed.
4. Split Data: The series is split into training and testing sets chronologically (e.g., 80% train, 20% test).
5. Auto Parameter Selection: Optimal (p, q) orders are selected by minimizing the Akaike Information Criterion (AIC).
6. Train Model: An ARIMA(p, d, q) model is trained, with sentiment as exogenous variables for SARIMAX.
7. Evaluate: The model is evaluated on the test set using walk-forward validation.
8. Inverse Transform: Predictions are inverse transformed back to the original price scale.
9. Calculate Metrics: Regression and classification metrics are calculated.
10. Analyze Feature Importance: Model coefficients are analyzed.
11. Save Model & Results.

**Parameters Used:**
- (p, d, q) orders: Automatically selected (p up to 3, q up to 3).
- Exogenous Variables: Daily Sentiment Score.
- Evaluation Test Size: 20% of data.
- Maximum Differencing Attempts: 2.

**Mathematical Formulation:**

$$
(1 - \sum_{i=1}^p \phi_i B^i) (1-B)^d y_t = (1 + \sum_{j=1}^q \theta_j B^j) \epsilon_t
$$

Where:
- $y_t$ is the time series at time $t$.
- $B$ is the backshift operator ($By_t = y_{t-1}$).
- $\phi_i$ are the autoregressive coefficients.
- $\theta_j$ are the moving average coefficients.
- $\epsilon_t$ is the error term.
- $d$ is the differencing order.

SARIMAX extends this by adding seasonal components and exogenous regressors.

**Evaluation Metrics:** RMSE, MAE, MAPE, R2 (Regression), Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity, Confusion Matrix (Classification).

**Feature Importance:** Analysis of the fitted model's coefficients and estimated variance.

---

## 2. LSTM (Long Short-Term Memory) Networks

**Introduction & Working Principles:**
- LSTMs are a type of Recurrent Neural Network (RNN) designed for sequential data like time series.
- They learn long-term dependencies, overcoming the vanishing gradient problem.
- LSTMs use gates (Forget, Input, Output) and a memory cell to regulate information flow.

**Algorithm/Steps:**
1. Load Data: Historical stock price data is loaded from MongoDB.
2. Feature Engineering: Technical indicators and sentiment scores are calculated.
3. Feature Selection: Relevant features are selected.
4. Scaling: Features are scaled (e.g., MinMaxScaler).
5. Create Sequences: Data is transformed into sequences for LSTM input.
6. Define Target Labels: Binary labels (Up/Down) are created for classification.
7. Split Data: Sequences are split into training and testing sets.
8. Build Model: A Keras sequential model with stacked LSTM layers is built.
9. Compile Model: Compiled with Adam optimizer and MSE loss.
10. Train Model: Trained with Early Stopping and Model Checkpointing.
11. Evaluate: Model predicts on test sequences.
12. Inverse Transform: Predictions are inverse transformed.
13. Calculate Metrics: Regression and classification metrics are calculated.
14. Analyze Feature Importance: SHAP values are calculated using DeepExplainer.
15. Save Model & Scalers.

**Parameters Used:**
- seq_length: Length of input sequences (default 60).
- lstm_units: Number of units in LSTM layers (default 60).
- dropout_rate: Dropout rate (default 0.2).
- epochs: Number of training epochs (default 75).
- batch_size: Batch size (default 32).
- patience: Patience for Early Stopping (default 15).
- Feature Windows: short_window (20), long_window (50).
- lag_days: Number of days for lagged features (default 1).
- Test Size: 20%.

**Mathematical Formulation (LSTM Cell):**

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) \\
C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \\
h_t &= o_t * \tanh(C_t)
\end{aligned}
$$

**Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity, Confusion Matrix (Classification).

**Feature Importance:** SHAP values using DeepExplainer.

---

## 3. SVM (Support Vector Machine) for Regression (SVR)

**Introduction & Working Principles:**
- SVM is a supervised learning model for classification and regression.
- **SVR:** Finds a hyperplane that best fits the data points within a specified error tolerance (epsilon).
- **Kernel Trick:** Uses kernel functions (RBF, linear, polynomial) to capture non-linear relationships.

**Algorithm/Steps:**
1. Load Data: Historical stock price data is loaded from MongoDB.
2. Prepare Data: Technical indicators, lagged features, and sentiment scores are included.
3. Define Target: The target is the next day's closing price.
4. Separate X and y: Features and target are separated.
5. Scaling: Features and target are scaled using StandardScaler.
6. Feature Selection: SelectKBest is used for feature selection.
7. Split Data: Data is split into training and testing sets.
8. Train Model: SVR is trained with hyperparameter tuning (GridSearchCV).
9. Evaluate: Model predicts on test features.
10. Inverse Transform: Predictions and actuals are inverse transformed.
11. Calculate Metrics: Regression and classification metrics are calculated.
12. Analyze Feature Importance: SHAP values using KernelExplainer.
13. Save Model & Scalers.

**Parameters Used:**
- look_back: Look-back period (default 60).
- num_features_to_select: Number of features (default 30).
- kernel: Kernel type (default 'rbf').
- C: Regularization parameter.
- gamma: Kernel coefficient.
- epsilon: Margin of tolerance.
- Test Size: 20%.

**Mathematical Formulation (SVR Objective):**

$$
\min_{w, b, \zeta, \zeta^*} \frac{1}{2} ||w||^2 + C \sum_{i=1}^m (\zeta_i + \zeta_i^*)
$$

Subject to:

$$
\begin{aligned}
y_i - f(x_i) &\leq \epsilon + \zeta_i \\
f(x_i) - y_i &\leq \epsilon + \zeta_i^* \\
\zeta_i, \zeta_i^* &\geq 0
\end{aligned}
$$

Where $w$ is the weight vector, $b$ is the bias, $\phi(x)$ is the mapped feature vector, $\epsilon$ is the margin, $C$ is the regularization parameter.

**Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Confusion Matrix, Classification Report (Classification).

**Feature Importance:** SHAP values using KernelExplainer.

---

## 4. XGBoost (Extreme Gradient Boosting)

**Introduction & Working Principles:**
- XGBoost is a gradient boosting algorithm known for performance and speed.
- It builds a sequence of decision trees, each correcting errors of the previous ones.
- Uses gradient descent to minimize the loss function and includes regularization.

**Algorithm/Steps:**
1. Load Data: Historical stock price data is loaded from MongoDB.
2. Prepare Data: Technical indicators, lagged features, and sentiment scores are included.
3. Define Target: The target is the next day's closing price.
4. Separate X and y: Features and target are separated.
5. Scaling: Features and target are scaled using StandardScaler.
6. Feature Selection: SelectKBest is used for feature selection.
7. Split Data: Data is split into training and testing sets.
8. Train Model: XGBoost Regressor is trained with hyperparameter tuning (GridSearchCV).
9. Evaluate: Model predicts on test features.
10. Inverse Transform: Predictions and actuals are inverse transformed.
11. Calculate Metrics: Regression and classification metrics are calculated.
12. Analyze Feature Importance: XGBoost's built-in feature_importances_ and SHAP values using TreeExplainer.
13. Save Model & Scalers.

**Parameters Used:**
- look_back: Look-back period (default 60).
- num_features_to_select: Number of features (default 30).
- n_estimators: Number of boosting rounds.
- max_depth: Maximum tree depth.
- learning_rate: Step size shrinkage.
- subsample: Fraction of training instances per tree.
- colsample_bytree: Fraction of features per tree.
- gamma: Minimum loss reduction for split.
- reg_alpha (L1) and reg_lambda (L2): Regularization terms.
- Test Size: 20%.

**Mathematical Formulation:**

$$
\hat{y}_i = \sum_{k=1}^K f_k(x_i)
$$

Regularized objective function:

$$
obj(\theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$

Where $l(y_i, \hat{y}_i)$ is the loss function, $\Omega(f_k)$ is the regularization term.

**Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Confusion Matrix, Classification Report (Classification).

**Feature Importance:** XGBoost's built-in feature importance and SHAP values using TreeExplainer.

---

These descriptions align with the project report and code, providing a comprehensive view of how each model is applied to stock market prediction, including data processing, training, and evaluation procedures.


## 🗂️ Project Structure Detailed Explanation

### Root Directory (/home/skylap/Downloads/stockmarketprediction)

- **.env, .env.example**: Environment variable files. .env holds sensitive information like API keys and database credentials (and should be in .gitignore), while .env.example serves as a template.
- **.gitignore**: Specifies intentionally untracked files that Git should ignore (like .env, venv, __pycache__, db.sqlite3).
- **database_schema.json**: Contains a JSON representation of the database schema, generated by extract_schemas.py.
- **db.sqlite3**: The default SQLite database file used by Django for development.
- **extract_schemas.py**: A Python script used to generate the database_schema.json file.
- **finance-chat.*.gguf**: Model files in the GGUF format for a local Large Language Model (LLM) related to finance chat functionality.
- **manage.py**: The Django command-line utility for administrative tasks.
- **Procfile**: Used by deployment platforms like Heroku to declare process types and commands to run.
- **README.md**: Contains information about the project, setup instructions, etc.
- **requirements.txt**: Lists the Python package dependencies required for the project.
- **runtime.txt**: Used by deployment platforms to specify the required Python runtime version.
- **run_vllm_server.py**: A script to run a VLLM (Very Large Language Model) server, using the .gguf files.
- **stock_symbols.csv**: A CSV file containing stock symbols for importing initial stock data.
- **train_jobs.log**: A log file recording the status and output of model training jobs.

### stockmarketprediction/ (Django Project Directory)

Core configuration files for the Django project:
- **__init__.py**: Marks the directory as a Python package.
- **asgi.py**: Configuration for running the project using ASGI, suitable for asynchronous features like WebSockets.
- **mongodb_router.py**: Handles routing database queries, enabling the project to use MongoDB alongside the default SQL database.
- **settings.py**: The main Django project settings file (database configuration, installed apps, middleware, etc.).
- **urls.py**: The main URL configuration, routing incoming requests to the appropriate Django apps or views.
- **wsgi.py**: Configuration for running the project using WSGI, the standard for synchronous Python web applications.

### stocks/ (Django App Directory)

The main application handling stock-related functionality:
- **admin.py, admin_views.py**: Register models with and extend the Django admin interface.
- **ai_service.py**: Handles interactions with AI/ML models for analysis or predictions.
- **apps.py**: Configuration specific to the stocks app.
- **forms.py**: Defines Django forms used for user input.
- **groq_service.py**: Integration with the Groq API for fast LLM inference.
- **management/commands/**: Custom Django management commands (import_stocks.py, update_commodities.py, etc.).
- **migrations/**: Database migration files generated by Django based on model changes.
- **models_*.py**: Database models and extensions for the stocks app.
- **mongo_utils.py**: Utility functions for interacting with MongoDB.
- **news_service.py**: Handles fetching and processing news articles related to stocks.
- **sentiment_service.py**: Analyzes news or text data to determine sentiment.
- **services.py**: Business logic for stock-related operations.
- **static/**: Static files (CSS, JavaScript, images) specific to the stocks app.
- **templates/**: HTML templates for the stocks app.
- **templatetags/**: Custom Django template tags and filters.
- **tests.py**: Unit tests for the stocks app.
- **urls.py**: URL configuration specific to the stocks app.
- **views.py**: View functions that handle requests and return responses.

### templates/ (Project-Level Templates)

Templates used across the entire project:
- **registration/**: Templates for user authentication (login.html, register.html).

### train-model/ (Model Training Directory)

Scripts and data related to training machine learning models:
- **AAPL/**: Data and trained models specific to Apple stock.
- **train-*.py**: Python scripts for training different types of models (ARIMA, LSTM, SVM, XGBoost).
- **train.py**: Master training script orchestrating model training.
- **training_batch.json**: Configuration for batch training jobs.

### Other Directories

- **models/**: Intended for ML model files.
- **staticfiles/**: Created by Django's collectstatic command, gathers all static files into a single location for serving.
- **venv/**: Python virtual environment containing isolated project dependencies.