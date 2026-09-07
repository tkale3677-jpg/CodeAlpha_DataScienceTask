# ============================================================
# TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

file_path = "Unemployment in India.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("UNEMPLOYMENT ANALYSIS WITH PYTHON")
print("=" * 60)

# ------------------------------------------------------------
# 2. BASIC DATA EXPLORATION
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ------------------------------------------------------------
# 3. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = df.columns.str.strip()

print("\nCleaned Column Names:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

print("\nDataset Shape After Cleaning:")
print(df.shape)

# ------------------------------------------------------------
# 5. CONVERT DATE COLUMN
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

# Remove invalid dates
df = df.dropna(subset=["Date"])

# ------------------------------------------------------------
# 6. CREATE YEAR AND MONTH COLUMNS
# ------------------------------------------------------------

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()

print("\nData after date processing:")
print(df.head())

# ------------------------------------------------------------
# 7. OVERALL UNEMPLOYMENT TREND
# ------------------------------------------------------------

monthly_unemployment = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment["Date"],
    monthly_unemployment["Estimated Unemployment Rate (%)"],
    marker="o"
)

plt.title("Unemployment Rate Trend in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. YEAR-WISE UNEMPLOYMENT RATE
# ------------------------------------------------------------

yearly_rate = (
    df.groupby("Year")["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\nYear-wise Average Unemployment Rate:")
print(yearly_rate)

plt.figure(figsize=(10, 5))

yearly_rate.plot(
    kind="bar"
)

plt.title("Average Unemployment Rate by Year")
plt.xlabel("Year")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. STATE/REGION-WISE ANALYSIS
# ------------------------------------------------------------

region_rate = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Regions with Highest Average Unemployment:")
print(region_rate.head(10))

plt.figure(figsize=(12, 7))

region_rate.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Regions with Highest Average Unemployment")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. LOWEST UNEMPLOYMENT REGIONS
# ------------------------------------------------------------

print("\nTop 10 Regions with Lowest Average Unemployment:")
print(region_rate.tail(10))

# ------------------------------------------------------------
# 11. COVID-19 IMPACT ANALYSIS
# ------------------------------------------------------------

# Before COVID: 2019 and early 2020
before_covid = df[
    df["Date"] < "2020-03-01"
]

# COVID