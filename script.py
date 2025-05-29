import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred # For easily downloading FRED data
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit # For proper time series cross-validation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
FRED_API_KEY = 'a092661752fc53217660813b15ad4a10' # !!! REPLACE WITH YOUR ACTUAL FRED API KEY !!!
START_DATE = '2005-01-01' # Last 20 years from roughly 2025
END_DATE = '2025-05-01' # Up to current available data

# FRED Series IDs for key factors
# Home Price Index (Dependent Variable)
HPI_SERIES_ID = 'CSUSHPISA' # S&P CoreLogic Case-Shiller U.S. National Home Price Index

# Independent Variables (Factors)
FACTORS = {
    'Mortgage30Year': 'MORTGAGE30US', # 30-Year Fixed Rate Mortgage Average
    'UnemploymentRate': 'UNRATE',       # Unemployment Rate
    'CPI_AllItems': 'CPIAUCSL',        # Consumer Price Index, All Items
    'NewHousingStarts': 'HOUST',       # New Privately-Owned Housing Units Started: Total Units
    'RealGDP': 'GDPC1',                # Real Gross Domestic Product (Quarterly, will need handling)
    'ConsumerSentiment': 'UMCSENT',    # University of Michigan: Consumer Sentiment Index
    'Population': 'POPTHM'             # Total Population (Monthly)
}

# --- 1. Helper Function to Download Data from FRED ---
def get_fred_data(api_key, series_id, start_date, end_date):
    """Downloads time series data from FRED."""
    fred = Fred(api_key=api_key)
    df = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    return df

# --- 2. Data Loading and Initial Preprocessing Structure ---
def load_and_preprocess_data(api_key, hpi_id, factor_ids, start_date, end_date):
    """
    Loads data from FRED, merges, and preprocesses.
    """
    print("--- Loading Home Price Index Data ---")
    hpi_data = get_fred_data(api_key, hpi_id, start_date, end_date)
    hpi_data = hpi_data.to_frame(name='HomePriceIndex')
    hpi_data.index = pd.to_datetime(hpi_data.index)
    hpi_data = hpi_data.resample('MS').mean() # Resample to monthly start, taking mean for consistency

    print("\n--- Loading Factor Data ---")
    all_factors_data = pd.DataFrame(index=hpi_data.index) # Use HPI index as base

    for name, series_id in factor_ids.items():
        print(f"Loading: {name} ({series_id})")
        factor_series = get_fred_data(api_key, series_id, start_date, end_date)
        factor_series.index = pd.to_datetime(factor_series.index)
        # Resample to monthly, forward fill for quarterly data and then fill any remaining NaNs
        factor_series = factor_series.resample('MS').mean().ffill()
        all_factors_data[name] = factor_series

    # Merge all data
    df = pd.merge(hpi_data, all_factors_data, left_index=True, right_index=True, how='inner')

    # Handle any remaining NaN values (e.g., at the very beginning of the series if data starts later)
    # A common strategy is to forward fill, then backward fill for initial NaNs.
    df = df.ffill().bfill()

    print("\n--- Initial Data Snapshot ---")
    print(df.head())
    print(df.info())
    print(f"\nData shape: {df.shape}")

    # Calculate year-over-year change for Home Price Index as a target for some models
    df['HomePriceIndex_YoY'] = df['HomePriceIndex'].pct_change(12) * 100
    # Drop first 12 months for YoY to avoid NaNs
    df = df.dropna(subset=['HomePriceIndex_YoY'])

    print("\n--- Data with YoY Change ---")
    print(df.head(15)) # Show more rows to see YoY
    return df

# --- 3. Basic Feature Engineering ---
def create_features(df):
    """Creates additional features for modeling."""
    print("\n--- Creating Features ---")
    # Example: Lagged features for independent variables (e.g., impact after 1-3 months)
    for col in FACTORS.keys():
        for lag in [1, 3, 6]: # Example lags
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Example: Year-over-year change for some factors
    for col in ['Mortgage30Year', 'UnemploymentRate', 'NewHousingStarts']:
        df[f'{col}_YoY'] = df[col].pct_change(12) * 100

    # Drop rows with NaNs created by lagging/YoY changes
    df = df.dropna()
    print(df.head())
    print(f"Data shape after feature engineering: {df.shape}")
    return df

# --- 4. Exploratory Data Analysis (EDA) Structure ---
def perform_eda(df):
    """Performs exploratory data analysis."""
    print("\n--- Performing EDA ---")

    # Plot Home Price Index over time
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['HomePriceIndex'], label='US National Home Price Index')
    plt.title('US National Home Price Index (2005-2025)')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Home Price Index YoY change
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['HomePriceIndex_YoY'], label='US National Home Price Index YoY Change', color='orange')
    plt.title('US National Home Price Index Year-over-Year Change (2005-2025)')
    plt.xlabel('Date')
    plt.ylabel('YoY Change (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot key factors
    fig, axes = plt.subplots(len(FACTORS), 1, figsize=(12, 4 * len(FACTORS)))
    axes = axes.flatten()
    for i, (name, _) in enumerate(FACTORS.items()):
        axes[i].plot(df.index, df[name])
        axes[i].set_title(f'{name} over Time')
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

    # Correlation Matrix
    plt.figure(figsize=(12, 10))
    # Select relevant columns for correlation (original factors + HPI YoY)
    corr_cols = [col for col in FACTORS.keys()] + ['HomePriceIndex_YoY']
    # Include some key engineered features if desired
    # corr_cols += [f'Mortgage30Year_lag_1', 'NewHousingStarts_YoY'] # Example
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Key Factors and Home Price Index YoY Change')
    plt.tight_layout()
    plt.show()

    print("\n--- EDA Complete ---")

# --- 5. Model Building (SARIMAX Example) ---
def build_and_evaluate_sarimax(df):
    """Builds and evaluates a SARIMAX model."""
    print("\n--- Building SARIMAX Model ---")

    # Target variable: HomePriceIndex_YoY
    # Exogenous variables: Select key factors (original or engineered)
    exog_vars = [
        'Mortgage30Year', 'UnemploymentRate', 'CPI_AllItems',
        'NewHousingStarts', 'ConsumerSentiment', 'Population',
        'Mortgage30Year_lag_1', 'NewHousingStarts_lag_3' # Using some engineered features
    ]

    # Ensure all selected exogenous variables exist and handle any missing values in them
    # before splitting, though create_features and load_and_preprocess_data should handle most.
    # For SARIMAX, you typically use differenced data for the endogenous variable to ensure stationarity.
    # However, SARIMAX handles differencing internally with the 'd' parameter.
    # We will model the raw HomePriceIndex for 'd' to take care of it, or HomePriceIndex_YoY if we believe it's already stationary.
    # Let's use HomePriceIndex_YoY as it often has better stationarity properties for direct modeling.

    # Drop any remaining NaNs after selecting exog_vars
    df_sarimax = df.dropna(subset=exog_vars + ['HomePriceIndex_YoY'])
    if df_sarimax.empty:
        print("Error: DataFrame for SARIMAX is empty after dropping NaNs for selected features.")
        return

    y = df_sarimax['HomePriceIndex_YoY']
    X = df_sarimax[exog_vars]

    # Time-based split: e.g., 80% train, 20% test
    train_size = int(len(df_sarimax) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]

    print(f"SARIMAX Training data shape: {y_train.shape}, Test data shape: {y_test.shape}")

    # SARIMAX model parameters (p, d, q) (P, D, Q, S)
    # These often require some trial and error, or auto_arima for automated selection.
    # For a start, a simple (1,0,1) for non-seasonal and (1,0,0,12) for seasonal is a common initial guess
    # if you expect yearly seasonality. If using YoY, seasonality might be less pronounced.
    order = (1, 0, 1)      # (AR order, Differencing order, MA order)
    seasonal_order = (0, 0, 0, 0) # (Seasonal AR, Seasonal Differencing, Seasonal MA, Seasonality period)
                               # Set to (0,0,0,0) if you believe YoY already removes seasonality,
                               # or if no strong seasonality is expected.

    try:
        model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False) # disp=False suppresses full optimization output
        print(model_fit.summary())

        # Make predictions
        start_index = len(y_train)
        end_index = len(df_sarimax) - 1
        predictions = model_fit.predict(start=start_index, end=end_index, exog=X_test)
        predictions.index = y_test.index # Align index for comparison

        # Evaluation
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(f"\nSARIMAX Model Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R-squared: {r2:.2f}")

        # Plot predictions vs actual
        plt.figure(figsize=(14, 7))
        plt.plot(y_train.index, y_train, label='Train Home Price Index YoY')
        plt.plot(y_test.index, y_test, label='Actual Home Price Index YoY', color='orange')
        plt.plot(predictions.index, predictions, label='SARIMAX Predictions', color='green', linestyle='--')
        plt.title('SARIMAX Model: Home Price Index YoY Prediction vs Actual')
        plt.xlabel('Date')
        plt.ylabel('YoY Change (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Interpretation of coefficients (if significant)
        print("\n--- SARIMAX Model Interpretation (Exogenous Coefficients) ---")
        print(model_fit.pvalues[exog_vars]) # P-values for coefficients
        print(model_fit.params[exog_vars]) # Coefficients

        # Discuss coefficients:
        # For example, a positive coefficient for 'Mortgage30Year' would be counter-intuitive.
        # Ensure signs align with economic intuition (e.g., higher rates -> lower prices).
        # Sometimes, non-stationarity or multicollinearity can affect coefficient signs.
        # Looking at p-values helps identify statistically significant factors.

    except Exception as e:
        print(f"Error during SARIMAX model fitting or prediction: {e}")
        print("Consider adjusting SARIMAX orders (p,d,q)(P,D,Q,S) or checking data for issues.")


# --- 6. Model Building (Machine Learning Example - RandomForestRegressor) ---
def build_and_evaluate_ml_model(df):
    """Builds and evaluates a Machine Learning model (RandomForestRegressor)."""
    print("\n--- Building Machine Learning Model (RandomForestRegressor) ---")

    # Target variable
    y = df['HomePriceIndex_YoY']

    # Features: Original factors + engineered features (lags, YoY of factors)
    # It's crucial to use only lagged features of 'y' and current/lagged 'X' variables
    # to avoid data leakage (using future information).
    features = [col for col in df.columns if col not in ['HomePriceIndex', 'HomePriceIndex_YoY']]

    # Filter out any features that might have NaNs introduced by earlier steps if not fully handled
    features = [f for f in features if f in df.columns and not df[f].isnull().any()]

    X = df[features]

    # Drop any remaining NaNs after selecting features
    X_ml = X.dropna()
    y_ml = y.loc[X_ml.index] # Align y with X after dropping NaNs

    if X_ml.empty:
        print("Error: DataFrame for ML model is empty after dropping NaNs for selected features.")
        return

    # Time-based split for ML models
    train_size = int(len(X_ml) * 0.8)
    X_train, X_test = X_ml[:train_size], X_ml[train_size:]
    y_train, y_test = y_ml[:train_size], y_ml[train_size:]

    print(f"ML Training data shape: {y_train.shape}, Test data shape: {y_test.shape}")

    # Initialize and train the RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    predictions_series = pd.Series(predictions, index=y_test.index)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\nRandomForestRegressor Model Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Plot predictions vs actual
    plt.figure(figsize=(14, 7))
    plt.plot(y_train.index, y_train, label='Train Home Price Index YoY')
    plt.plot(y_test.index, y_test, label='Actual Home Price Index YoY', color='orange')
    plt.plot(predictions_series.index, predictions_series, label='RandomForest Predictions', color='purple', linestyle='--')
    plt.title('RandomForestRegressor Model: Home Price Index YoY Prediction vs Actual')
    plt.xlabel('Date')
    plt.ylabel('YoY Change (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Feature Importance
    print("\n--- RandomForestRegressor Feature Importance ---")
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)
    print(sorted_importances.head(15)) # Print top 15 features

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances.head(15).values, y=sorted_importances.head(15).index, palette='viridis')
    plt.title('Top 15 Feature Importances in RandomForestRegressor')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Discuss implications of top features:
    # For example, if 'Mortgage30Year_lag_1' is highly important, it implies interest rates
    # from the previous month significantly influence current home price changes.
    # If 'UnemploymentRate' is high, it suggests job market stability is key.

# --- Main Execution Flow ---
if __name__ == "__main__":
    # 1. Load and Preprocess Data
    df_raw = load_and_preprocess_data(FRED_API_KEY, HPI_SERIES_ID, FACTORS, START_DATE, END_DATE)

    # 2. Create Features
    df_processed = create_features(df_raw.copy()) # Use a copy to avoid modifying raw df directly

    # 3. Perform EDA
    perform_eda(df_processed)

    # 4. Build and Evaluate SARIMAX Model
    build_and_evaluate_sarimax(df_processed.copy())

    # 5. Build and Evaluate ML Model
    build_and_evaluate_ml_model(df_processed.copy())

    print("\n--- Data Science Model Building Task Complete ---")
    print("Remember to replace 'YOUR_FRED_API_KEY' with your actual key!")
    print("Further steps include hyperparameter tuning, more advanced feature engineering, and robust economic interpretation.")