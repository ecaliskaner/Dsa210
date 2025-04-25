import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, zscore, pearsonr
from scipy.stats import norm

# Step 1: Understand the Problem
# Research question: Does weather (especially rain) significantly affect traffic in Istanbul?

# Step 2: Import and Inspect the Data
weather = pd.read_csv("Weather1.csv", parse_dates=["date"])
traffic = pd.read_csv("traffic_index.csv", parse_dates=["trafficindexdate"])

# Fix datetime format mismatch
weather["date"] = pd.to_datetime(weather["date"]).dt.date.astype(str)
traffic["trafficindexdate"] = pd.to_datetime(traffic["trafficindexdate"]).dt.date.astype(str)
traffic.rename(columns={"trafficindexdate": "date"}, inplace=True)

# Merge datasets
df = pd.merge(weather, traffic, on="date", how="outer")
df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)

# Step 3: Handle Missing Data
print("Missing values:\n", df.isnull().sum())
df["average_traffic_index"] = df["average_traffic_index"].interpolate()
df["prcp"] = df["prcp"].fillna(0)

# Step 4: Explore Data Characteristics
print("\nWeather Summary:\n", df["prcp"].describe())
print("\nTraffic Summary:\n", df["average_traffic_index"].describe())

# Step 5: Data Transformation
df["rainy_day"] = df["prcp"] > 0

# Step 6: Visualize Data Relationships
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["average_traffic_index"], label="Traffic Index")
plt.title("Traffic Index Over Time")
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Traffic Index")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["prcp"], color="blue", label="Precipitation (prcp)")
plt.title("Precipitation Over Time")
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Precipitation")
plt.legend()
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))
corr = df[["average_traffic_index", "prcp"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Traffic Index and Precipitation")
plt.show()

# Step 7: Handling Outliers
# Visualize boxplots
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["average_traffic_index"])
plt.title("Traffic Index Outliers")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df["prcp"])
plt.title("Precipitation Outliers")
plt.show()

# Step 8: T-test (Statistical Test)
rainy = df[df["rainy_day"] == True]["average_traffic_index"].dropna()
dry = df[df["rainy_day"] == False]["average_traffic_index"].dropna()

t_stat, p_val = ttest_ind(rainy, dry, equal_var=False)
print(f"\nT-statistic: {t_stat:.2f}")
print(f"P-value: {p_val:.5f}")
if p_val < 0.05:
    print("✅ Significant difference in traffic between rainy and dry days.")
else:
    print("❌ No significant difference found.")

# Visualize T-test
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)
plt.plot(x, y, color='black', label='t-distribution')
plt.axvline(x=t_stat, color='red', linestyle='--', label=f'T-statistic: {t_stat:.2f}')
plt.title("T-test Visualization: Rainy vs Dry Traffic")
plt.legend()
plt.tight_layout()
plt.show()

# Step 8 (continued): Pearson Correlation Value
corr_coef, p_corr = pearsonr(df["prcp"], df["average_traffic_index"])
print(f"\nPearson correlation between precipitation and traffic index: {corr_coef:.2f} (p={p_corr:.5f})")
