# ==========================================================
# TASK 1: IRIS FLOWER CLASSIFICATION
# ==========================================================

# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------------
# 2. Load Iris Dataset
# ----------------------------------------------------------

iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

# Add flower species name
df["species"] = df["target"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------------------------------------
# 3. Data Visualization
# ----------------------------------------------------------

# Pairplot
sns.pairplot(
    df,
    hue="species",
    vars=iris.feature_names
)
plt.suptitle("Iris Flower Feature Analysis", y=1.02)
plt.show()

# ----------------------------------------------------------
# 4. Prepare Input and Output
# ----------------------------------------------------------

X = df[iris.feature_names]
y = df["target"]

# ----------------------------------------------------------
# 5. Split Dataset into Training and Testing Data
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)

# ----------------------------------------------------------
# 6. Feature Scaling
# ----------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# 7. Create Machine Learning Model
# ----------------------------------------------------------

model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train_scaled, y_train)

# ----------------------------------------------------------
# 8. Make Predictions
# ----------------------------------------------------------

y_pred = model.predict(X_test_scaled)

# ----------------------------------------------------------
# 9. Evaluate Model
# ----------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# ----------------------------------------------------------
# 10. Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix - Iris Classification")
plt.show()

# ----------------------------------------------------------
# 11. Test Model with New Flower
# ----------------------------------------------------------

# Example flower measurements:
# [Sepal Length, Sepal Width, Petal Length, Petal Width]

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

# Scale the new data
new_flower_scaled = scaler.transform(new_flower)

# Predict
prediction = model.predict(new_flower_scaled)

# Display result
print("\n===================================")
print("NEW FLOWER PREDICTION")
print("===================================")

print("Predicted Species:", iris.target_names[prediction[0]])

# Prediction probability
probability = model.predict_proba(new_flower_scaled)

print("\nPrediction Probabilities:")

for species, prob in zip(iris.target_names, probability[0]):
    print(species, ":", round(prob * 100, 2), "%")

# ----------------------------------------------------------
# 12. Final Result
# ----------------------------------------------------------

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================")
print("The Iris flower classification model was trained")
print("and tested successfully.")
print("Model Accuracy:", round(accuracy * 100, 2), "%")