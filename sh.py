import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv("creditcard.csv")

# Data summary
print("Dataset Shape:", df.shape)
print(df.head())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0].sample(n=len(fraud), random_state=1)
balanced_df = pd.concat([fraud, normal]).sample(frac=1, random_state=1)

print("Balanced Dataset Shape:", balanced_df.shape)
print(balanced_df['Class'].value_counts())

# Feature and target split
X = balanced_df.drop('Class', axis=1)
y = balanced_df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=[X.columns[i] for i in indices], y=importances[indices])
plt.xticks(rotation=90)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
