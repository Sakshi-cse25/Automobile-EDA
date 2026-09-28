import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# EXPLORATORY DATA ANALYSIS ON AUTOMOBILE DATASET
# ============================================================

# 1. LOAD DATASET
# UCI Automobile Dataset
file_path = "imports-85.data"

df = pd.read_csv(
    file_path,
    header=None,
    names=[
        "symboling", "normalized_losses", "make", "fuel_type", "aspiration",
        "num_of_doors", "body_style", "drive_wheels", "engine_location",
        "wheel_base", "length", "width", "height", "curb_weight",
        "engine_type", "num_of_cylinders", "engine_size", "fuel_system",
        "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
        "city_mpg", "highway_mpg", "price"
    ],
    na_values="?"
)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. DATA QUALITY CHECK
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

numeric_columns = [
    "symboling", "normalized_losses", "wheel_base", "length", "width",
    "height", "curb_weight", "engine_size", "bore", "stroke",
    "compression_ratio", "horsepower", "peak_rpm", "city_mpg",
    "highway_mpg", "price"
]

categorical_columns = [
    "make", "fuel_type", "aspiration", "num_of_doors", "body_style",
    "drive_wheels", "engine_location", "engine_type",
    "num_of_cylinders", "fuel_system"
]

print("\nNumerical Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)

# 3. CLEANING
before_rows = len(df)

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove duplicate rows
df = df.drop_duplicates()

# Price is the target descriptive variable, so rows without price are removed.
df = df.dropna(subset=["price"])

# Median imputation for remaining numerical missing values
for col in numeric_columns:
    if col != "price":
        df[col] = df[col].fillna(df[col].median())

# Mode imputation for categorical columns
for col in categorical_columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Standardize categorical text
for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Remove impossible values if present
df = df[df["price"] > 0]
df = df[df["engine_size"] > 0]

after_rows = len(df)

print("\nRows Before Cleaning:", before_rows)
print("Rows After Cleaning:", after_rows)
print("Rows Removed:", before_rows - after_rows)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicates After Cleaning:", df.duplicated().sum())

# 4. DESCRIPTIVE STATISTICS
print("\nNumerical Descriptive Statistics:")
print(df[numeric_columns].describe())

print("\nCategorical Descriptive Statistics:")
print(df[categorical_columns].describe())

# Price outlier detection using IQR
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

price_outliers = df[
    (df["price"] < lower_bound) | (df["price"] > upper_bound)
]

print("\nPrice Outliers:", len(price_outliers))
print("Outlier bounds:", lower_bound, "to", upper_bound)

# Outliers are retained because they may represent genuine expensive cars.

# Correlation with price
correlation = df[
    ["price", "engine_size", "horsepower", "curb_weight",
     "length", "width", "wheel_base", "city_mpg", "highway_mpg"]
].corr()["price"].sort_values(ascending=False)

print("\nCorrelation with Price:")
print(correlation)

# ============================================================
# 5. VISUAL EDA
# ============================================================

# Chart 1: Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=20, kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

print("Observation: Car prices are concentrated in the lower-to-middle price range, with fewer high-priced cars.")

# Chart 2: Cars by Manufacturer
plt.figure(figsize=(10, 5))
df["make"].value_counts().plot(kind="bar")
plt.title("Number of Cars by Manufacturer")
plt.xlabel("Manufacturer")
plt.ylabel("Number of Cars")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Observation: The dataset contains different numbers of cars for each manufacturer, so manufacturer counts are uneven.")

# Chart 3: Fuel Type
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="fuel_type")
plt.title("Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()

print("Observation: Gasoline cars are much more common than diesel cars in this dataset.")

# Chart 4: Price by Fuel Type
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="fuel_type", y="price")
plt.title("Car Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

print("Observation: The price distributions of gasoline and diesel cars differ, although their ranges overlap.")

# Chart 5: Engine Size vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="engine_size", y="price")
plt.title("Engine Size vs Car Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

print("Observation: Larger engine sizes generally appear alongside higher car prices.")

# Chart 6: Horsepower vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="horsepower", y="price")
plt.title("Horsepower vs Car Price")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

print("Observation: Higher horsepower generally corresponds to higher prices, with some variation.")

# Chart 7: Correlation Heatmap
plt.figure(figsize=(10, 7))
corr_cols = [
    "price", "engine_size", "horsepower", "curb_weight",
    "length", "width", "wheel_base", "city_mpg", "highway_mpg"
]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.show()

print("Observation: Price has positive relationships with engine size, horsepower, curb weight and vehicle dimensions.")

# Chart 8: Average Price by Manufacturer
avg_price_make = (
    df.groupby("make")["price"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))
avg_price_make.plot(kind="bar")
plt.title("Average Car Price by Manufacturer")
plt.xlabel("Manufacturer")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Observation: Average prices vary substantially across manufacturers.")

# ============================================================
# 6. KEY FINDINGS
# ============================================================

print("\n========== KEY FINDINGS ==========")

print("1. The dataset contains", len(df), "cars after cleaning.")
print("2. The median car price is:", df["price"].median())
print("3. The most common manufacturer is:", df["make"].mode()[0])
print("4. The most common fuel type is:", df["fuel_type"].mode()[0])
print("5. Engine size and horsepower show positive relationships with price.")
print("6. Average car price varies across manufacturers.")
print("7. Price contains some high-value observations that were retained because they may represent genuine vehicles.")

print("\n========== LIMITATIONS ==========")
print("- The dataset contains only 205 original records.")
print("- The data represents automobiles from a particular historical period.")
print("- Some categories have relatively few observations.")
print("- Correlation does not prove causation.")

print("\n========== CONCLUSION ==========")
print("The exploratory analysis shows meaningful relationships between automobile price and")
print("features such as engine size, horsepower, curb weight and manufacturer.")
print("Data cleaning improved consistency while preserving potentially meaningful observations.")
