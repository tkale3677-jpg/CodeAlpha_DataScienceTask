# ============================================================
# TASK 3: CAR PRICE PREDICTION WITH MACHINE LEARNING
# ============================================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "car data.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("CAR PRICE PREDICTION")
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

df.columns = df.columns.str.strip()

print("\nCleaned Column Names:")
print(df.columns.tolist())


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 5. REMOVE DUPLICATE ROWS
# ============================================================

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape After Removing Duplicates:")
print(df.shape)


# ============================================================
# 6. IDENTIFY TARGET COLUMN
# ============================================================

# Most common Car Price dataset uses Selling_Price
target_column = "Selling_Price"

if target_column not in df.columns:

    # Try alternate column names
    possible_targets = [
        "selling_price",
        "Selling Price",
        "selling price",
        "Price",
        "price"
    ]

    found_target = None

    for column in possible_targets:
        if column in df.columns:
            found_target = column
            break

    if found_target:
        target_column = found_target

    else:
        raise ValueError(
            "Selling price column not found. "
            "Check your dataset column names."
        )


print("\nTarget Column:", target_column)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ============================================================
# 8. ENCODE CATEGORICAL VARIABLES
# ============================================================

df_model = df.copy()

label_encoders = {}

for column in df_model.select_dtypes(
    include="object"
).columns:

    encoder = LabelEncoder()

    df_model[column] = encoder.fit_transform(
        df_model[column].astype(str)
    )

    label_encoders[column] = encoder


print("\nData After Encoding:")
print(df_model.head())


# ============================================================
# 9. SEPARATE FEATURES AND TARGET
# ============================================================

X = df_model.drop(
    columns=[target_column]
)

y = df_model[target_column]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(target_column)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 11. TRAIN RANDOM FOREST REGRESSION MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


print("\nModel Training Completed!")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 13. MODEL EVALUATION
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

print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

print(
    "R2 Score Percentage:",
    round(r2 * 100, 2),
    "%"
)


# ============================================================
# 14. ACTUAL VS PREDICTED PRICE
# ============================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(comparison.head(10))


# ============================================================
# 15. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title(
    "Actual vs Predicted Car Prices"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# ============================================================
# 17. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Feature Importance in Car Price Prediction"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.show()


# ============================================================
# 18. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    df_model.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. PREDICT A SAMPLE CAR
# ============================================================

sample_car = X_test.iloc[[0]]

sample_prediction = model.predict(
    sample_car
)

print("\n" + "=" * 60)
print("SAMPLE CAR PRICE PREDICTION")
print("=" * 60)

print(
    "Predicted Price:",
    round(sample_prediction[0], 2)
)


# ============================================================
# 20. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "The Random Forest Regression model was trained "
    "to predict car prices."
)

print(
    "Model R2 Score:",
    round(r2 * 100, 2),
    "%"
)