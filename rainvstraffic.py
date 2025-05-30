# Weather Impact on Istanbul Traffic Analysis
# Author: [Your Name]
# Date: [Current Date]
# Description: This script analyzes the relationship between weather conditions
#              and traffic patterns in Istanbul using various data science techniques.

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

# Visualize time series patterns
plt.figure(figsize=(15, 6))
sns.lineplot(data=df, x='date', y='average_traffic_index', hue='is_rainy', alpha=0.6)
plt.title('Traffic Index Over Time - Rainy vs Non-Rainy Days')
plt.xticks(rotation=45)
plt.legend(title='Is Rainy', labels=['No Rain', 'Rain'])
plt.tight_layout()
plt.show()
