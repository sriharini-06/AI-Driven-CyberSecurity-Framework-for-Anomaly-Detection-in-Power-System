# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')  # Enable UTF-8 encoding

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Split features and target
X = df.drop(columns=["snort_alert", "Use_Case"])
y = df["snort_alert"]

# Feature Scaling (not always necessary for Decision Trees, but keeping it for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Decision Tree Fine-Tuned Parameters
params = {
    'criterion': [ 'entropy'],
    'max_depth': [10],
    'min_samples_split': [2]
}

tree_model = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(tree_model, params, cv=StratifiedKFold(n_splits=5), scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# Best Model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
print("\n🔹 Best Parameters:", grid.best_params_)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔹 Decision Tree Accuracy: {accuracy:.6f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Confusion Matrix with specified style
dt_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={"size": 30, "fontname": "Times New Roman"},
            cbar=False)
plt.xlabel('Predicted', fontsize=40, fontname="Times New Roman")
plt.ylabel('True', fontsize=40, fontname="Times New Roman")
plt.xticks(fontsize=30, fontname="Times New Roman")
plt.yticks(fontsize=30, fontname="Times New Roman")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(best_model, "models/decision_tree_model.pkl")
print("\n✅ Model saved as 'decision_tree_model.pkl'")
