# ============================================================
# TASK 4: SALES PREDICTION USING PYTHON
# ============================================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "sales_data.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("SALES PREDICTION USING PYTHON")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. BASIC DATA EXPLORATION
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

print("\nCleaned Column Names:")
print(df.columns.tolist())


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# Fill missing numerical values
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


# Fill missing categorical values
categorical_columns = df.select_dtypes(
    exclude=np.number
).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset Shape After Cleaning:")
print(df.shape)


# ============================================================
# 6. FIND SALES COLUMN
# ============================================================

possible_sales_columns = [
    "Sales",
    "sales",
    "Sales_Value",
    "sales_value",
    "Total_Sales",
    "total_sales",
    "Revenue",
    "revenue"
]

sales_column = None

for column in possible_sales_columns:
    if column in df.columns:
        sales_column = column
        break


if sales_column is None:

    raise ValueError(
        "Sales column was not found. "
        "Please check your CSV column names."
    )


print("\nTarget Sales Column:")
print(sales_column)


# ============================================================
# 7. DATA VISUALIZATION
# ============================================================

# Select numerical columns
numeric_df = df.select_dtypes(
    include=np.number
)

if sales_column in numeric_df.columns:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df[sales_column],
        kde=True
    )

    plt.title("Sales Distribution")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Variables")

plt.tight_layout()
plt.show()


# ============================================================
# 9. ENCODE CATEGORICAL VARIABLES
# ============================================================

df_model = df.copy()

label_encoders = {}

for column in df_model.select_dtypes(
    include="object"
).columns:

    # Do not encode date-like information here
    encoder = LabelEncoder()

    df_model[column] = encoder.fit_transform(
        df_model[column].astype(str)
    )

    label_encoders[column] = encoder


print("\nData After Encoding:")
print(df_model.head())


# ============================================================
# 10. SEPARATE FEATURES AND TARGET
# ============================================================

X = df_model.drop(
    columns=[sales_column]
)

y = df_model[sales_column]


print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget:")
print(sales_column)


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# 12. TRAIN LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed!")


# ============================================================
# 13. PREDICT SALES
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    "Mean Absolute Error:",
    round(mae, 2)
)

print(
    "Mean Squared Error:",
    round(mse, 2)
)

print(
    "Root Mean Squared Error:",
    round(rmse, 2)
)

print(
    "R2 Score:",
    round(r2, 4)
)

print(
    "R2 Score Percentage:",
    round(r2 * 100, 2),
    "%"
)


# ============================================================
# 15. ACTUAL VS PREDICTED SALES
# ============================================================

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted Sales:")
print(comparison.head(10))


# ============================================================
# 16. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title(
    "Actual vs Predicted Sales"
)

plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 17. ADVERTISING FEATURE ANALYSIS
# ============================================================

# Find advertising-related columns
advertising_columns = []

for column in df.columns:

    column_lower = column.lower()

    if (
        "advert" in column_lower
        or "marketing" in column_lower
        or "tv" == column_lower
        or "radio" == column_lower
        or "newspaper" == column_lower
    ):
        advertising_columns.append(column)


print("\nAdvertising-related Columns:")
print(advertising_columns)


# Create graphs for advertising columns
for column in advertising_columns:

    if column in numeric_df.columns:

        plt.figure(figsize=(8, 5))

        plt.scatter(
            df[column],
            df[sales_column],
            alpha=0.6
        )

        plt.xlabel(column)
        plt.ylabel(sales_column)

        plt.title(
            f"{column} vs Sales"
        )

        plt.grid(True)

        plt.tight_layout()
        plt.show()


# ============================================================
# 18. FEATURE IMPORTANCE / COEFFICIENTS
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

feature_importance["Absolute_Impact"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    by="Absolute_Impact",
    ascending=False
)

print("\nFeature Impact:")
print(feature_importance)


# ============================================================
# 19. FEATURE IMPACT GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Absolute_Impact",
    y="Feature"
)

plt.title(
    "Feature Impact on Sales Prediction"
)

plt.xlabel("Absolute Coefficient")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# ============================================================
# 20. SALES DISTRIBUTION BY CATEGORICAL VARIABLES
# ============================================================

for column in df.select_dtypes(
    include="object"
).columns:

    if df[column].nunique() <= 15:

        plt.figure(figsize=(10, 6))

        sns.boxplot(
            data=df,
            x=column,
            y=sales_column
        )

        plt.title(
            f"Sales Distribution by {column}"
        )

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


# ============================================================
# 21. BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

print(
    "1. Advertising expenditure can influence sales performance."
)

print(
    "2. Feature analysis helps identify variables strongly "
    "associated with sales."
)

print(
    "3. Actual vs predicted values help evaluate the "
    "quality of the prediction model."
)

print(
    "4. Businesses can use sales predictions to improve "
    "marketing and resource planning."
)

print(
    "5. Advertising budgets can be optimized based on "
    "historical sales patterns."
)


# ============================================================
# 22. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("TASK 4 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "Model R2 Score:",
    round(r2 * 100, 2),
    "%"
)