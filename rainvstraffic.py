# Weather Impact on Istanbul Traffic Analysis
# Description: This script analyzes the relationship between weather conditions
# and traffic patterns in Istanbul using various data science techniques.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, zscore, pearsonr
from scipy.stats import norm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define visualization functions before model training
def plot_combined_weather_effects():
    """Visualize the combined effects of weather conditions on traffic in 3D"""
    # Create a single figure for 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    
    # Create 3D scatter plot
    scatter = ax3d.scatter(df['temperature_avg'],
                          df['wind_speed_kmh'],
                          df['average_traffic_index'],
                          c=df['precipitation_mm'],
                          cmap='YlOrRd',
                          alpha=0.6,
                          s=50)  # Increased marker size
    
    # Customize the plot
    ax3d.set_xlabel('Temperature (°C)', labelpad=10)
    ax3d.set_ylabel('Wind Speed (km/h)', labelpad=10)
    ax3d.set_zlabel('Traffic Index', labelpad=10)
    ax3d.set_title('3D Weather Effects on Traffic', pad=20, size=14)
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Precipitation (mm)', rotation=270, labelpad=15)
    
    # Rotate the view for better visualization
    ax3d.view_init(elev=20, azim=45)
    
    # Add grid
    ax3d.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance():
    """Visualize feature importance from different models"""
    # Prepare feature names
    feature_names = weather_factors + ['is_weekend', 'is_rainy', 
                                     'rain_temperature_interaction',
                                     'rain_wind_interaction',
                                     'weekend_rain_interaction']
    
    # Get feature importance from Random Forest
    rf_importance = rf_grid.best_estimator_.feature_importances_
    
    # Get feature importance from Gradient Boosting
    gb_importance = gb_grid.best_estimator_.feature_importances_
    
    # Get coefficients from Linear Regression (standardized)
    lr = LinearRegression()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr.fit(X_scaled, y)
    lr_importance = np.abs(lr.coef_)
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(feature_names))
    width = 0.25
    
    plt.bar(x - width, rf_importance, width, label='Random Forest', alpha=0.7)
    plt.bar(x, gb_importance, width, label='Gradient Boosting', alpha=0.7)
    plt.bar(x + width, lr_importance / lr_importance.max(), width, 
            label='Linear Regression (Normalized)', alpha=0.7)
    
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance Comparison Across Models')
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print numerical feature importance
    print("\nFeature Importance Rankings:")
    importance_data = {
        'Random Forest': rf_importance,
        'Gradient Boosting': gb_importance,
        'Linear Regression': lr_importance / lr_importance.max()
    }
    
    importance_df = pd.DataFrame(importance_data, index=feature_names)
    print("\nAverage Feature Importance Across Models:")
    print(importance_df.mean(axis=1).sort_values(ascending=False).round(3))

# Step 1: Data Import and Preprocessing
# Load weather and traffic data from CSV files
weather = pd.read_csv("Weather1.csv", parse_dates=["date"])
traffic = pd.read_csv("traffic_index.csv", parse_dates=["trafficindexdate"])

# Rename columns to be more descriptive
weather = weather.rename(columns={
    'tavg': 'temperature_avg',
    'prcp': 'precipitation_mm',
    'wspd': 'wind_speed_kmh',
    'wdir': 'wind_direction_degrees',
    'pres': 'pressure_hpa',
    'tsun': 'sunshine_hours'
})

# Standardize date formats for merging
weather["date"] = pd.to_datetime(weather["date"])
traffic["trafficindexdate"] = pd.to_datetime(traffic["trafficindexdate"])
traffic.rename(columns={"trafficindexdate": "date"}, inplace=True)

# Merge datasets on date
df = pd.merge(weather, traffic, on="date", how="outer")
df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)

# Add weekday features (moved after merge to ensure they're based on final dates)
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
df['day_of_week'] = df['date'].dt.day_name()

# Handle missing values
print("Missing values before cleaning:\n", df.isnull().sum())
df["average_traffic_index"] = df["average_traffic_index"].interpolate()
df["precipitation_mm"] = df["precipitation_mm"].fillna(0)
df["wind_speed_kmh"] = df["wind_speed_kmh"].fillna(df["wind_speed_kmh"].mean())
df["pressure_hpa"] = df["pressure_hpa"].fillna(df["pressure_hpa"].mean())
print("\nMissing values after cleaning:\n", df.isnull().sum())

# Step 2: Feature Engineering
# Create categorical variables for analysis
df['temperature_category'] = pd.qcut(df['temperature_avg'], q=5, 
                                   labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Very Warm'])
df['wind_category'] = pd.qcut(df['wind_speed_kmh'], q=5, 
                             labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
df['is_rainy'] = df['precipitation_mm'] > 0

# Step 3: Statistical Analysis
# Define weather factors with clear names
weather_factors = [
    'temperature_avg',
    'precipitation_mm',
    'wind_speed_kmh',
    'pressure_hpa'
]

# Calculate correlations between weather factors and traffic
correlations = {}
p_values = {}

for factor in weather_factors:
    corr, p_val = pearsonr(df[factor], df['average_traffic_index'])
    correlations[factor] = corr
    p_values[factor] = p_val

print("\nCorrelations with Traffic Index:")
for factor in weather_factors:
    print(f"{factor}: r={correlations[factor]:.3f} (p={p_values[factor]:.5f})")

# Weekday Analysis
print("\n=== Weekday vs Weekend Traffic Analysis ===")
weekday_traffic = df[~df['is_weekend']]['average_traffic_index']
weekend_traffic = df[df['is_weekend']]['average_traffic_index']

print("\nWeekday vs Weekend Comparison:")
print(f"Average weekday traffic: {weekday_traffic.mean():.2f}")
print(f"Average weekend traffic: {weekend_traffic.mean():.2f}")
print(f"Difference (Weekday - Weekend): {weekday_traffic.mean() - weekend_traffic.mean():.2f}")

# Statistical test for weekday vs weekend
t_stat_week, p_val_week = ttest_ind(weekday_traffic, weekend_traffic)
print(f"\nT-test Results (Weekday vs Weekend):")
print(f"T-statistic: {t_stat_week:.2f}")
print(f"P-value: {p_val_week:.5f}")

# Daily patterns
daily_traffic = df.groupby('day_of_week')['average_traffic_index'].agg(['mean', 'std']).round(2)
daily_traffic.index = pd.CategoricalIndex(daily_traffic.index, 
                                        categories=['Monday', 'Tuesday', 'Wednesday', 
                                                  'Thursday', 'Friday', 'Saturday', 'Sunday'])
daily_traffic = daily_traffic.sort_index()
print("\nAverage Traffic by Day of Week:")
print(daily_traffic)

# Visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='day_of_week', y='average_traffic_index')
plt.title('Traffic Index Distribution by Day of Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='day_of_week', y='average_traffic_index', hue='is_rainy')
plt.title('Traffic Index by Day of Week and Rain Condition')
plt.xticks(rotation=45)
plt.legend(title='Is Rainy', labels=['No Rain', 'Rain'])
plt.tight_layout()
plt.show()

# Step 4: Visualization
# Create correlation heatmap
plt.figure(figsize=(10, 8))
weather_traffic_corr = df[weather_factors + ['average_traffic_index']].corr()
sns.heatmap(weather_traffic_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap: Weather Factors vs Traffic')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create scatter plots for each weather factor
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, factor in enumerate(weather_factors):
    sns.regplot(data=df, x=factor, y='average_traffic_index', ax=axes[idx],
                scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    axes[idx].set_title(f'Traffic vs {factor}')
    axes[idx].set_xlabel(factor.replace('_', ' ').title())
    axes[idx].set_ylabel('Traffic Index')

plt.tight_layout()
plt.show()

# Analyze categorical relationships
# Temperature categories
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='temperature_category', y='average_traffic_index', hue='is_rainy')
plt.title('Traffic Index by Temperature Category and Rain Condition')
plt.xticks(rotation=45)
plt.legend(title='Is Rainy', labels=['No Rain', 'Rain'])
plt.tight_layout()
plt.show()

# Wind categories
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='wind_category', y='average_traffic_index', hue='is_rainy')
plt.title('Traffic Index by Wind Category and Rain Condition')
plt.xticks(rotation=45)
plt.legend(title='Is Rainy', labels=['No Rain', 'Rain'])
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nAverage Traffic Index by Temperature Category and Rain Condition:")
print(df.groupby(['temperature_category', 'is_rainy'])['average_traffic_index'].agg(['mean', 'count', 'std']).round(2))

print("\nAverage Traffic Index by Wind Category and Rain Condition:")
print(df.groupby(['wind_category', 'is_rainy'])['average_traffic_index'].agg(['mean', 'count', 'std']).round(2))

# Update machine learning features
X = df[weather_factors + ['is_weekend']].copy()  # Add is_weekend to features
X['is_rainy'] = df['is_rainy'].astype(int)
X['rain_temperature_interaction'] = X['is_rainy'] * X['temperature_avg']
X['rain_wind_interaction'] = X['is_rainy'] * X['wind_speed_kmh']
X['weekend_rain_interaction'] = X['is_weekend'].astype(int) * X['is_rainy']  # Convert boolean to int
y = df['average_traffic_index']

# Step 5: Advanced Machine Learning Analysis
print("\n=== Advanced Machine Learning Analysis ===")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(kernel='rbf')
}

# Function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    print(f"Explained Variance Score: {ev:.2f}")
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'ev': ev}

# Dictionary to store results
results = {}

# Create pipeline with scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R² score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = evaluate_model(y_test, y_pred, name)

# Hyperparameter tuning for best performing models
print("\n=== Hyperparameter Tuning ===")

# Random Forest tuning
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), 
                      rf_params, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

print("\nBest Random Forest Parameters:", rf_grid.best_params_)
y_pred_rf = rf_grid.predict(X_test_scaled)
rf_results = evaluate_model(y_test, y_pred_rf, "Tuned Random Forest")

# Gradient Boosting tuning
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 4, 5]
}

gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), 
                      gb_params, cv=5, scoring='r2', n_jobs=-1)
gb_grid.fit(X_train_scaled, y_train)

print("\nBest Gradient Boosting Parameters:", gb_grid.best_params_)
y_pred_gb = gb_grid.predict(X_test_scaled)
gb_results = evaluate_model(y_test, y_pred_gb, "Tuned Gradient Boosting")

# Compare model performances
model_comparison = pd.DataFrame(results).T
print("\nModel Comparison:")
print(model_comparison)

# Visualize model comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=model_comparison.reset_index(), x='index', y='r2')
plt.title('R² Score Comparison Across Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final Analysis: Compare rainy vs non-rainy days
rainy_traffic = df[df['is_rainy']]['average_traffic_index']
non_rainy_traffic = df[~df['is_rainy']]['average_traffic_index']

print("\nTraffic Comparison - Rainy vs Non-Rainy Days:")
print(f"Average traffic on rainy days: {rainy_traffic.mean():.2f}")
print(f"Average traffic on non-rainy days: {non_rainy_traffic.mean():.2f}")
print(f"Difference: {rainy_traffic.mean() - non_rainy_traffic.mean():.2f}")

# Statistical significance test
t_stat, p_val = ttest_ind(rainy_traffic, non_rainy_traffic)
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_val:.5f}")

# First figure: Time series
plt.figure(figsize=(15, 6))
# Filter data to start from when precipitation data is available and after May 28, 2022
df_clean = df.dropna(subset=['precipitation_mm'])
df_clean = df_clean[df_clean['date'] >= '2022-05-28']
sns.lineplot(data=df_clean, x='date', y='average_traffic_index', 
             hue='is_rainy', alpha=0.6)
plt.title('Traffic Index Over Time - Rainy vs Non-Rainy Days')
plt.xlabel('Date')
plt.ylabel('Traffic Index')
plt.legend(title='Is Rainy', labels=['No Rain', 'Rain'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# After the time series plot and before the distribution plot, add:
print("\n=== Hypothesis Testing ===")
print("Null Hypothesis (H0): There is no significant difference in traffic index between rainy and non-rainy days")
print("Alternative Hypothesis (H1): There is a significant difference in traffic index between rainy and non-rainy days")
print("\nTest Method: Independent Two-Sample T-Test")
print(f"Sample Size:")
print(f"- Rainy days: {len(rainy_traffic)}")
print(f"- Non-rainy days: {len(non_rainy_traffic)}")
print(f"\nDescriptive Statistics:")
print(f"- Rainy days mean: {rainy_traffic.mean():.2f} ± {rainy_traffic.std():.2f} (SD)")
print(f"- Non-rainy days mean: {non_rainy_traffic.mean():.2f} ± {non_rainy_traffic.std():.2f} (SD)")
print(f"\nTest Results:")
print(f"- T-statistic: {t_stat:.2f}")
print(f"- P-value: {p_val:.10f}")
print(f"- Mean difference: {rainy_traffic.mean() - non_rainy_traffic.mean():.2f}")
print("\nConclusion:")
if p_val < 0.05:
    print("Reject the null hypothesis (p < 0.05)")
    print("There is strong statistical evidence that rain significantly affects traffic patterns")
else:
    print("Fail to reject the null hypothesis (p >= 0.05)")
    print("There is not enough statistical evidence to conclude that rain affects traffic patterns")

# Create visualization of hypothesis test results
plt.figure(figsize=(15, 6))

# Create kernel density estimation for both distributions
rainy_kde = sns.kdeplot(data=rainy_traffic, label='Rainy Days', 
                        color='red', linewidth=2)
non_rainy_kde = sns.kdeplot(data=non_rainy_traffic, label='Non-Rainy Days', 
                           color='blue', linewidth=2)

# Add vertical lines for means
plt.axvline(x=rainy_traffic.mean(), color='red', linestyle='--', alpha=0.5,
            label=f'Rainy Mean: {rainy_traffic.mean():.2f}')
plt.axvline(x=non_rainy_traffic.mean(), color='blue', linestyle='--', alpha=0.5,
            label=f'Non-Rainy Mean: {non_rainy_traffic.mean():.2f}')

# Add significance annotation
if p_val < 0.05:
    y_max = plt.gca().get_ylim()[1]
    plt.text(0.5 * (rainy_traffic.mean() + non_rainy_traffic.mean()), y_max * 0.95,
             f'p = {p_val:.2e}', ha='center')
    if p_val < 0.001:
        sig_text = '***'
    elif p_val < 0.01:
        sig_text = '**'
    else:
        sig_text = '*'
    plt.text(0.5 * (rainy_traffic.mean() + non_rainy_traffic.mean()), y_max * 0.9,
             sig_text, ha='center', fontsize=12)

plt.title(f'Traffic Distribution: Rainy vs Non-Rainy Days\nT-statistic: {t_stat:.2f}, p-value: {p_val:.2e}')
plt.xlabel('Traffic Index')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Step 6: Future Prediction Functionality
print("\n=== Future Traffic Prediction ===")

def prepare_prediction_features(weather_forecast):
    """
    Prepare features for prediction from weather forecast data
    
    Parameters:
    weather_forecast: DataFrame with columns:
        - date
        - temperature_avg
        - precipitation_mm
        - wind_speed_kmh
        - pressure_hpa
    
    Returns:
    Prepared feature matrix ready for prediction
    """
    X_pred = weather_forecast[weather_factors].copy()
    X_pred['is_weekend'] = weather_forecast['date'].dt.dayofweek.isin([5, 6])
    X_pred['is_rainy'] = weather_forecast['precipitation_mm'] > 0
    X_pred['rain_temperature_interaction'] = X_pred['is_rainy'] * X_pred['temperature_avg']
    X_pred['rain_wind_interaction'] = X_pred['is_rainy'] * X_pred['wind_speed_kmh']
    X_pred['weekend_rain_interaction'] = X_pred['is_weekend'].astype(int) * X_pred['is_rainy']
    return X_pred

def predict_traffic(weather_forecast, model=None, scaler=None):
    """
    Predict traffic index based on weather forecast
    
    Parameters:
    weather_forecast: DataFrame with weather forecast data
    model: Trained model (if None, uses the tuned Gradient Boosting model)
    scaler: Fitted StandardScaler (if None, uses the existing scaler)
    
    Returns:
    DataFrame with predictions and confidence intervals
    """
    if model is None:
        model = gb_grid.best_estimator_
    
    if scaler is None:
        scaler = StandardScaler().fit(X)
    
    # Prepare features
    X_pred = prepare_prediction_features(weather_forecast)
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make predictions
    predictions = model.predict(X_pred_scaled)
    
    # Calculate prediction intervals using residuals
    y_train_pred = model.predict(X_train_scaled)
    residuals = y_train - y_train_pred
    residual_std = np.std(residuals)
    
    # Create prediction DataFrame
    results_df = pd.DataFrame({
        'date': weather_forecast['date'],
        'predicted_traffic_index': predictions,
        'lower_bound': predictions - 1.96 * residual_std,
        'upper_bound': predictions + 1.96 * residual_std
    })
    
    return results_df

# Example usage with sample forecast data
if __name__ == "__main__":
    # Create sample forecast data for next 7 days
    last_date = df['date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
    
    # Create two forecast scenarios: normal and rainy
    # Normal scenario
    normal_forecast = pd.DataFrame({
        'date': forecast_dates,
        'temperature_avg': np.random.normal(20, 5, 7),  # Sample temperatures around 20°C
        'precipitation_mm': np.random.uniform(0, 0.1, 7),  # Very little to no rain
        'wind_speed_kmh': np.random.normal(15, 5, 7),  # Sample wind speeds around 15 km/h
        'pressure_hpa': np.random.normal(1013, 5, 7)  # Sample pressure around 1013 hPa
    })
    
    # Rainy scenario (same conditions but with rain)
    rainy_forecast = pd.DataFrame({
        'date': forecast_dates,
        'temperature_avg': normal_forecast['temperature_avg'],  # Keep same temperature
        'precipitation_mm': np.random.uniform(5, 15, 7),  # Add significant rain (5-15mm)
        'wind_speed_kmh': normal_forecast['wind_speed_kmh'],  # Keep same wind
        'pressure_hpa': normal_forecast['pressure_hpa']  # Keep same pressure
    })
    
    # Make predictions for both scenarios
    normal_predictions = predict_traffic(normal_forecast)
    rainy_predictions = predict_traffic(rainy_forecast)
    
    print("\nPredicted Traffic Index for Next 7 Days (Normal Conditions):")
    print(normal_predictions.round(2))
    
    print("\nPredicted Traffic Index for Next 7 Days (Rainy Conditions):")
    print(rainy_predictions.round(2))
    
    # Visualize predictions for both scenarios
    plt.figure(figsize=(12, 6))
    
    # Plot normal conditions
    plt.plot(normal_predictions['date'], normal_predictions['predicted_traffic_index'], 
             'b-', label='Predicted Traffic (Normal)', alpha=0.7)
    plt.fill_between(normal_predictions['date'], 
                     normal_predictions['lower_bound'],
                     normal_predictions['upper_bound'],
                     alpha=0.1, color='b')
    
    # Plot rainy conditions
    plt.plot(rainy_predictions['date'], rainy_predictions['predicted_traffic_index'], 
             'r-', label='Predicted Traffic (Rainy)', alpha=0.7)
    plt.fill_between(rainy_predictions['date'], 
                     rainy_predictions['lower_bound'],
                     rainy_predictions['upper_bound'],
                     alpha=0.1, color='r')
    
    plt.title('Traffic Predictions: Normal vs Rainy Conditions')
    plt.xlabel('Date')
    plt.ylabel('Traffic Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print comparative insights
    print("\nComparative Prediction Insights:")
    print("Normal Conditions:")
    print(f"Average predicted traffic index: {normal_predictions['predicted_traffic_index'].mean():.2f}")
    print(f"Maximum predicted traffic: {normal_predictions['predicted_traffic_index'].max():.2f}")
    print(f"Minimum predicted traffic: {normal_predictions['predicted_traffic_index'].min():.2f}")
    
    print("\nRainy Conditions:")
    print(f"Average predicted traffic index: {rainy_predictions['predicted_traffic_index'].mean():.2f}")
    print(f"Maximum predicted traffic: {rainy_predictions['predicted_traffic_index'].max():.2f}")
    print(f"Minimum predicted traffic: {rainy_predictions['predicted_traffic_index'].min():.2f}")
    
    print(f"\nAverage traffic increase due to rain: "
          f"{(rainy_predictions['predicted_traffic_index'].mean() - normal_predictions['predicted_traffic_index'].mean()):.2f}")

    print("\n=== Combined Weather Effects Analysis ===")
    plot_combined_weather_effects()
    
    print("\n=== Feature Importance Analysis ===")
    plot_feature_importance()
