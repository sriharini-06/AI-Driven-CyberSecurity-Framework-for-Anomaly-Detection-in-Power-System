import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Enable UTF-8 encoding

# Load your dataset
df = pd.read_csv("cleaned_dataset.csv")
X = df.drop(['snort_alert', 'Use_Case'], axis=1)
y = df['Use_Case']

X = X.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'var_smoothing': [1e-12,1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
}

grid = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

rf_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['UC1', 'UC2','UC3','UC4'], yticklabels=['UC1', 'UC2','UC3','UC4'],annot_kws={"size": 30, "fontname": "Times New Roman"},cbar=False)
plt.xlabel('Predicted',fontsize=40,fontname="Times New Roman")
plt.ylabel('True',fontsize=40,fontname="Times New Roman")
plt.xticks(fontsize=30,fontname="Times New Roman")
plt.yticks(fontsize=30,fontname="Times New Roman")
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\classification\ml",'gnb_multi.png'),dpi=300)
plt.show()

#save model
joblib.dump(best_model, "models/gnb_multi_model.pkl")
print("\n✅ Model saved as 'gnb_multi_model.pkl'")