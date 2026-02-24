import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')
# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Split features and target
X = df.drop(columns=["snort_alert","Use_Case"])
y = df["Use_Case"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# SVM Fine-Tuned Parameters
params = {
    'C': [100],
    'kernel': ['rbf'],
    'gamma': ['auto']
}

svm_model = SVC()
grid = GridSearchCV(svm_model, params, cv=StratifiedKFold(n_splits=5), scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# Best Model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n🔹 Best Parameters:", grid.best_params_)
print(f"\n🔹 SVM Accuracy:{accuracy:.6f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

rf_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['UC1', 'UC2','UC3',"UC4"], yticklabels=['UC1', 'UC2','UC3',"UC4"],annot_kws={"size": 30, "fontname": "Times New Roman"},cbar=False)
plt.xlabel('Predicted',fontsize=40,fontname="Times New Roman")
plt.ylabel('True',fontsize=40,fontname="Times New Roman")
plt.xticks(fontsize=30,fontname="Times New Roman")
plt.yticks(fontsize=30,fontname="Times New Roman")
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\classification\ml",'svm_multi.png'),dpi=300)
plt.show()

#save models
joblib.dump(best_model, "models/svm_multi_model.pkl")
print("\n✅ Model saved as 'svm_multi_model.pkl'")