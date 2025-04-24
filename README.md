# Machine Learning Models Used for Stock Market Prediction

This project employs four distinct machine learning algorithms for predicting stock market price movement direction: **ARIMA**, **LSTM**, **SVM**, and **XGBoost**. Each model is chosen for its specific strengths in handling time-series data, capturing sequential dependencies, modeling non-linear relationships, and incorporating exogenous factors like sentiment. The models are trained on historical stock data and sentiment analysis derived from financial news articles, leveraging various technical indicators as features.

---

## Feature Importance: SHAP (SHapley Additive exPlanations)

Across several models in this project (LSTM, SVM, XGBoost), SHAP is used to analyze feature importance. SHAP is a modern approach rooted in cooperative game theory (specifically Shapley values) to explain the output of any machine learning model.

The core idea behind SHAP is to calculate the contribution of each feature to the difference between the actual prediction for a specific instance and the average (or baseline) prediction across the dataset. It assigns an "importance value" (the SHAP value) to each feature for each individual prediction, indicating how much that feature pushed the prediction away from the baseline.

SHAP values offer several key benefits:
1.  **Local Interpretability:** They provide an explanation for *why* a specific prediction was made for a single data point.
2.  **Unified Measure:** SHAP values have consistent interpretations across different model types, making comparisons easier.
3.  **Global Interpretability:** SHAP values can be aggregated across many predictions to understand the overall importance of each feature for the model.

Different SHAP "Explainers" are used depending on the type of model being explained:
*   **TreeExplainer:** Optimized for tree-based models like XGBoost.
*   **DeepExplainer:** Designed for deep learning models like LSTMs.
*   **KernelExplainer:** A model-agnostic explainer that works for any model, often used for kernel methods like SVM or when a specific explainer isn't available.

In this project, SHAP helps to understand which technical indicators, lagged prices, or sentiment scores were most influential in the models' predictions for stock price movement.

---

## 1. ARIMA (Autoregressive Integrated Moving Average) / SARIMAX (Seasonal ARIMA with Exogenous Regressors)

**Introduction & Working Principles:**
- ARIMA is a statistical model specifically designed for time series forecasting. Its core principle is to capture the temporal dependencies and patterns within the data itself.
- **AR (Autoregressive):** This component models the relationship between a current observation and a number of *lagged* (past) observations. It assumes that the future value is a linear combination of past values. The parameter 'p' indicates how many past observations are included.
- **I (Integrated):** This component addresses non-stationarity in the time series, which means the statistical properties (like mean and variance) change over time (e.g., has a trend or seasonality). Differencing involves computing the difference between consecutive observations until the series becomes stationary. The parameter 'd' is the number of times differencing is applied.
- **MA (Moving Average):** This component models the relationship between a current observation and past *forecast errors* (residuals) from a moving average model. It uses the dependency between an observation and the error terms of previous forecasts. The parameter 'q' indicates how many past error terms are included.
- **SARIMAX:** This is an extension of ARIMA. The 'S' adds components to handle *seasonal patterns* (patterns that repeat regularly, like daily, weekly, or yearly). The 'X' allows the inclusion of *exogenous variables*, which are external factors that are not part of the main time series but can influence it. In this project, sentiment scores derived from financial news are used as these exogenous regressors, allowing the model to incorporate external market sentiment alongside historical price data patterns.

**Algorithm/Steps:**
1. Load Data: Historical stock price data is loaded, focusing on the target price field ('Close') and other relevant columns.
2. Feature Engineering (Optional): SARIMAX incorporates exogenous variables (sentiment scores). Technical indicators may be calculated.
3. Check/Make Stationary: The time series data is checked for stationarity using the Augmented Dickey-Fuller (ADF) test. Differencing is applied if needed.
4. Split Data: The series is split into training and testing sets chronologically (e.g., 80% train, 20% test).
5. Auto Parameter Selection: Optimal (p, d, q) orders (and seasonal orders for SARIMA/SARIMAX) are often selected automatically by minimizing criteria like the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC), which balance model fit and complexity.
6. Train Model: An ARIMA(p, d, q) model, potentially with seasonal components and using sentiment as exogenous variables for SARIMAX, is trained on the historical data.
7. Evaluate: The model is evaluated on the test set, often using walk-forward validation for time series, where the model is retrained or updated as new data becomes available.
8. Inverse Transform: Predictions are inverse transformed back to the original price scale if differencing was applied.
9. Calculate Metrics: Regression metrics (RMSE, MAE, etc.) and classification metrics (Accuracy, F1-Score, etc., derived from predicting direction) are calculated.
10. Analyze Feature Importance: While not a traditional feature importance method like tree-based models, the significance of the model's coefficients and estimated variance for the AR, I, MA, and exogenous terms can be analyzed to understand their contribution.
11. Save Model & Results.

**Parameters Used:**
- (p, d, q) orders: Automatically selected (e.g., p up to 3, q up to 3).
- Exogenous Variables: Daily Sentiment Score.
- Evaluation Test Size: 20% of data.
- Maximum Differencing Attempts: 2.

**Evaluation Metrics:** RMSE, MAE, MAPE, R2 (Regression), Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity, Confusion Matrix (Classification).

**Feature Importance:** Analysis of the fitted model's coefficients and estimated variance for the terms (AR, MA, Exogenous).

---

## 2. LSTM (Long Short-Term Memory) Networks

**Introduction & Working Principles:**
- LSTMs are a specialized type of Recurrent Neural Network (RNN) specifically designed to handle sequential data, making them well-suited for time series like stock prices.
- Unlike simple RNNs, LSTMs are capable of learning and remembering dependencies over *long sequences* of data, effectively mitigating the "vanishing gradient problem" which hinders simple RNNs from learning long-term patterns.
- The key to LSTM's power lies in its internal structure, called the **LSTM cell**. Each cell contains several interacting *gates* and a *memory cell* (or state).
    - **Memory Cell:** This acts like a conveyor belt, carrying information through the sequence. It can add or remove information as needed.
    - **Forget Gate:** This gate controls what information from the *previous* memory cell state should be *discarded*. It looks at the current input and the previous hidden state to make this decision.
    - **Input Gate:** This gate decides what *new information* from the current input and previous hidden state should be *stored* in the memory cell. It has two parts: one decides which values to update, and the other creates a vector of new candidate values.
    - **Output Gate:** This gate determines what the *output* of the LSTM cell will be. It's based on the current input, the previous hidden state, and the newly updated memory cell state. This output (the hidden state) is then passed to the next step in the sequence.
- By carefully regulating the flow of information through these gates, LSTMs can selectively remember or forget information over long periods, allowing them to capture complex temporal patterns crucial for stock prediction.

**Algorithm/Steps:**
1. Load Data: Historical stock price data is loaded from MongoDB.
2. Feature Engineering: Technical indicators and sentiment scores are calculated as potential features.
3. Feature Selection: Relevant features are selected to reduce noise and complexity.
4. Scaling: Features are scaled (e.g., using MinMaxScaler) to normalize their ranges, which is important for neural networks.
5. Create Sequences: Data is transformed into sequences (or windows) of a defined length (e.g., past 60 days) that the LSTM will process. The target is typically the value on the day *after* the sequence ends.
6. Define Target Labels: Binary labels (Up/Down for classification) or the next day's price (for regression) are created as the prediction target.
7. Split Data: Sequences are split into training and testing sets chronologically to prevent look-ahead bias.
8. Build Model: A neural network model is constructed, typically using a Keras sequential model with one or more stacked LSTM layers, followed by dense layers for output.
9. Compile Model: The model is configured with an optimizer (like Adam) and a loss function appropriate for the task (e.g., MSE for regression, Binary Crossentropy for classification).
10. Train Model: The model is trained on the training data. Techniques like Early Stopping (to prevent overfitting) and Model Checkpointing (to save the best performing model) are used.
11. Evaluate: The trained model predicts on the test sequences.
12. Inverse Transform: If scaling was applied, predictions are inverse transformed back to the original scale.
13. Calculate Metrics: Regression and classification metrics are calculated based on the predictions and actual values.
14. Analyze Feature Importance: While LSTMs don't have built-in feature importance, methods like SHAP (using DeepExplainer for deep learning models) can be used to estimate the contribution of input features to the predictions (as described in the SHAP section).
15. Save Model & Scalers.

**Parameters Used:**
- seq_length: Length of input sequences (default 60 days).
- lstm_units: Number of units in LSTM layers (default 60).
- dropout_rate: Dropout rate (default 0.2) to prevent overfitting.
- epochs: Maximum number of training epochs (default 75).
- batch_size: Number of samples per gradient update (default 32).
- patience: Patience for Early Stopping (default 15 epochs without improvement).
- Feature Windows: short_window (20 days), long_window (50 days) for technical indicators.
- lag_days: Number of days for lagged features (default 1).
- Test Size: 20%.

**Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity, Confusion Matrix (Classification).

**Feature Importance:** SHAP values using DeepExplainer (as described in the SHAP section).

---

## 3. SVM (Support Vector Machine) for Regression (SVR)

**Introduction & Working Principles:**
- Support Vector Machines (SVMs) are powerful supervised learning models used for both classification and regression tasks.
- **Support Vector Regression (SVR):** The goal of SVR is to find a function that best fits the data points while limiting the error for each point to be within a certain tolerance, specified by a parameter called epsilon ($\epsilon$). Instead of trying to minimize the squared error for *every* data point (like linear regression), SVR aims to find a "tube" of width $2\epsilon$ around the predicted function, such that as many data points as possible fall *inside* this tube. The model complexity is penalized, trying to keep the function as "flat" as possible. Errors that fall *outside* the $\epsilon$-tube are penalized.
- **Epsilon-Insensitive Loss:** SVR uses an epsilon-insensitive loss function, meaning errors smaller than $\epsilon$ are not penalized at all. The model focuses on the data points that are *outside* or *on the boundary* of the $\epsilon$-tube around the predicted function.
- **Support Vectors:** The data points that lie outside or on the boundary of this $\epsilon$-tube are called Support Vectors. These are the crucial points that define the position and shape of the regression function.
- **Regularization Parameter (C):** A parameter 'C' controls the trade-off between allowing errors larger than $\epsilon$ (points outside the tube) and the complexity of the function. A smaller C allows for a simpler function but potentially more errors outside the tube, while a larger C tries harder to keep errors within the tube, potentially leading to a more complex function and overfitting.
- **Kernel Trick:** SVR can model non-linear relationships by using kernel functions (like Radial Basis Function (RBF), linear, polynomial). The kernel trick implicitly maps the data into a higher-dimensional space where a linear separation (or regression plane) might be possible, without explicitly calculating the coordinates in that high-dimensional space. This allows SVR to capture complex non-linear patterns in the stock data.

**Algorithm/Steps:**
1. Load Data: Historical stock price data is loaded from MongoDB.
2. Prepare Data: Features are engineered to include technical indicators, lagged price/volume data, and sentiment scores.
3. Define Target: The target variable is typically the next day's closing price or a binary variable indicating price direction (Up/Down).
4. Separate X and y: Features (X) and the target variable (y) are separated.
5. Scaling: Features and the target variable are scaled (e.g., using StandardScaler) as SVR is sensitive to the scale of the data.
6. Feature Selection: Methods like SelectKBest (based on statistical tests) can be used to select a subset of the most relevant features.
7. Split Data: Data is split into training and testing sets.
8. Train Model: An SVR model is trained on the training data. Hyperparameter tuning (e.g., using GridSearchCV to find optimal values for C, gamma, epsilon, and kernel) is crucial for performance.
9. Evaluate: The trained model makes predictions on the test features.
10. Inverse Transform: Predictions and actuals are inverse transformed back to the original scale if scaling was applied to the target.
11. Calculate Metrics: Regression metrics (MSE, RMSE, etc.) are calculated for price prediction, and classification metrics (Accuracy, Confusion Matrix, etc.) are calculated if the task is direction prediction.
12. Analyze Feature Importance: SVR does not have a built-in feature importance method like tree models. SHAP values using KernelExplainer (suitable for kernel-based models like SVR) can be computed to understand feature contributions (as described in the SHAP section).
13. Save Model & Scalers.

**Parameters Used:**
- look_back: Look-back period for creating lagged features (default 60 days).
- num_features_to_select: Number of features to keep after selection (default 30).
- kernel: Kernel type ('rbf', 'linear', 'poly'). Default is 'rbf'.
- C: Regularization parameter.
- gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels.
- epsilon: Epsilon-tube parameter (margin of tolerance).
- Test Size: 20%.

**Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Confusion Matrix, Classification Report (Classification).

**Feature Importance:** SHAP values using KernelExplainer (as described in the SHAP section).

---

## 4. XGBoost (Extreme Gradient Boosting)

**Introduction & Working Principles:**
- XGBoost is an optimized and highly efficient implementation of the **Gradient Boosting** algorithm. It is widely recognized for its speed, performance, and ability to handle various data types and complexities.
- **Gradient Boosting:** This is an ensemble technique where multiple *weak learners* (typically decision trees) are combined sequentially to form a strong learner. Instead of building trees independently (like Random Forests), gradient boosting builds trees *one after the other*. Each new tree is trained to correct the *errors* (or residuals) made by the *cumulative model* of all previously trained trees.
- **Gradient Descent:** The process of minimizing the errors by adding new trees is guided by gradient descent on a chosen loss function (like Mean Squared Error for regression or LogLoss for classification).
- **Additive Model:** The final prediction in XGBoost is the *sum* of the predictions from all the individual trees in the ensemble.
- **XGBoost Enhancements:** XGBoost includes several advanced features that improve performance and robustness compared to standard gradient boosting:
    - **Regularization:** Includes L1 (Lasso) and L2 (Ridge) regularization terms in the objective function to penalize complex models and prevent overfitting.
    - **Parallel Processing:** Allows training to be parallelized across CPU cores.
    - **Handling Missing Values:** Can automatically handle missing data in the features.
    - **Tree Pruning:** Uses a 'max_depth' parameter and criteria based on the loss reduction to prune trees.
    - **Advanced Split Finding:** Employs efficient algorithms for finding the best split points in trees.
    - **Column and Row Subsampling:** Similar to Random Forests, it can sample features and data points for each tree, further reducing overfitting and increasing speed. This combination of features makes XGBoost particularly effective for structured data like the technical indicators and sentiment scores used in this project.

**Algorithm/Steps:**
1. Load Data: Historical stock price data is loaded from MongoDB.
2. Prepare Data: Features are engineered including technical indicators, lagged features, and sentiment scores.
3. Define Target: The target variable is the next day's closing price (regression) or price direction (classification).
4. Separate X and y: Features (X) and the target variable (y) are separated.
5. Scaling: Features and target are scaled using StandardScaler, although tree-based models are generally less sensitive to scaling than models like SVM or neural networks.
6. Feature Selection: SelectKBest or other methods can be used to select relevant features.
7. Split Data: Data is split into training and testing sets.
8. Train Model: An XGBoost Regressor (for price prediction) or Classifier (for direction prediction) is trained. Hyperparameter tuning (e.g., using GridSearchCV or RandomizedSearchCV) is used to optimize parameters like `n_estimators`, `max_depth`, `learning_rate`, `subsample`, etc.
9. Evaluate: The trained model makes predictions on the test features.
10. Inverse Transform: If scaling was applied to the target, predictions are inverse transformed.
11. Calculate Metrics: Regression and classification metrics are calculated.
12. Analyze Feature Importance: XGBoost provides built-in methods to assess feature importance based on how often features are used in splits or how much they contribute to the model's improvement. SHAP values (using TreeExplainer for tree-based models) offer a more detailed and interpretable view of feature contributions for individual predictions (as described in the SHAP section).
13. Save Model & Scalers.

**Parameters Used:**
- look_back: Look-back period for lagged features (default 60 days).
- num_features_to_select: Number of features (default 30).
- n_estimators: Number of boosting rounds (trees).
- max_depth: Maximum depth of the trees.
- learning_rate: Step size shrinkage used in updates to prevent overfitting.
- subsample: Fraction of training instances used per tree.
- colsample_bytree: Fraction of features used per tree.
- gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree.
- reg_alpha (L1) and reg_lambda (L2): Regularization terms on weights.
- Test Size: 20%.

**Evaluation Metrics:** MSE, RMSE, MAE, MAPE, R2 (Regression), Accuracy, Confusion Matrix, Classification Report (Classification).

**Feature Importance:** XGBoost's built-in feature importance and SHAP values using TreeExplainer (as described in the SHAP section).

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