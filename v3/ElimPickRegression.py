import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import PartialDependenceDisplay

# Load the data
data = pd.read_csv("worlds_data.csv")


# Define independent and dependent variables
independent_variables = [
    "blue_banners",
    "total_epa_rank",
]
dependent_variables = ["elim_pick"]

# Create a binary column for being picked before 3rd
data["Picked_Top3"] = data["elim_pick"].apply(lambda x: 1 if x <= 2 and x != -1 else 0)

# keep only 50% of the datapoints where elim_pick is -1, discard rest
sample = data[data["elim_pick"] == -1].sample(frac=0.4)
data = data.drop(data[data["elim_pick"] == -1].index)
data = pd.concat([data, sample])


# Handle missing values
print("Missing values before cleaning:")
print(data[independent_variables + ["Picked_Top3"]].isnull().sum())
data = data.dropna(subset=independent_variables + ["Picked_Top3"])
print("Missing values after cleaning:")
print(data[independent_variables + ["Picked_Top3"]].isnull().sum())

# Exploratory Data Analysis
plt.figure(figsize=(8, 5))
sns.histplot(data["norm_epa"], kde=True, bins=30, color="skyblue")
plt.title("Distribution of Normalized EPA")
plt.xlabel("Normalized EPA")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data["state_epa_percentile"], kde=True, bins=30, color="salmon")
plt.title("Distribution of State EPA Percentile")
plt.xlabel("State EPA Percentile")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data["blue_banners"], kde=True, bins=30, color="lightgreen")
plt.title("Distribution of Blue Banners")
plt.xlabel("Blue Banners")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="elim_pick", data=data, palette="viridis")
plt.title("Distribution of Pick Positions")
plt.xlabel("Pick Position")
plt.ylabel("Count")
plt.show()

sns.pairplot(
    data[independent_variables + ["Picked_Top3"]], hue="Picked_Top3", palette="Set1"
)
plt.show()

# Prepare data for modeling
X = data[independent_variables]
y = data["Picked_Top3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {roc_auc:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Top 3", "Top 3"],
    yticklabels=["Not Top 3", "Top 3"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred))

# Feature Importance
coefficients = pd.Series(model.coef_[0], index=independent_variables)
coefficients = coefficients.sort_values()

plt.figure(figsize=(8, 6))
coefficients.plot(kind="barh", color="teal")
plt.title("Feature Coefficients in Logistic Regression")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

# Visualization of Predicted Probabilities
norm_epa_range = np.linspace(data["norm_epa"].min(), data["norm_epa"].max(), 300)

total_epa_percentile_range = np.linspace(
    data["total_epa_percentile"].min(), data["total_epa_percentile"].max(), 300
)
total_epa_rank_range = np.linspace(
    data["total_epa_rank"].min(), data["total_epa_rank"].max(), 300
)

mean_state_epa_percentile = data["state_epa_percentile"].mean()
mean_blue_banners = data["blue_banners"].mean()

predict_data = pd.DataFrame(
    {
        "blue_banners": mean_blue_banners,
        "total_epa_rank": total_epa_rank_range,
    }
)

predict_data_scaled = scaler.transform(predict_data)
probabilities = model.predict_proba(predict_data_scaled)[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="total_epa_rank", y="Picked_Top3", data=data, alpha=0.3, label="Actual Data"
)
plt.plot(
    total_epa_rank_range,
    probabilities,
    color="red",
    label="Logistic Regression Fit",
)
plt.title("Probability of Being Picked Before 3rd vs. Normalized EPA")
plt.xlabel("EPA Percentile")
plt.ylabel("Probability of Being Picked Before 3rd")
plt.legend()
plt.show()

# Partial Dependence Plot
features = independent_variables
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    model, X_train_scaled, features, feature_names=independent_variables, ax=ax
)
plt.show()
