# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')  # Enable UTF-8 encoding

# === Load dataset ===
df = pd.read_csv("cleaned_dataset.csv")

# === Split features and target ===
X = df.drop(columns=["snort_alert", "Use_Case"])
y = df["Use_Case"]

# === Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Random Forest Hyperparameter Tuning ===
params = {
    'n_estimators': [200],
    'max_depth': [10],
    'min_samples_split': [5],
    'min_samples_leaf': [1]
}

rf_model = RandomForestClassifier()
grid = GridSearchCV(rf_model, params, cv=StratifiedKFold(n_splits=5), scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# === Best Model ===
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# === Evaluation ===
print("\n🔹 Best Parameters:", grid.best_params_)
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Random Forest Accuracy: {accuracy:.6f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Confusion Matrix Plot ===
rf_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['UC1', 'UC2', 'UC3', 'UC4'],
            yticklabels=['UC1', 'UC2', 'UC3', 'UC4'],
            annot_kws={"size": 30, "fontname": "Times New Roman"},
            cbar=False)
plt.xlabel('Predicted', fontsize=40, fontname="Times New Roman", fontweight="bold")
plt.ylabel('True', fontsize=40, fontname="Times New Roman", fontweight="bold")
plt.xticks(fontsize=30, fontname="Times New Roman")
plt.yticks(fontsize=30, fontname="Times New Roman")
plt.tight_layout()
#plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\classification\ml", 'random_forest_multi.png'), dpi=300)
plt.show()

# === Save model ===
#joblib.dump(best_model, "models/random_forest_multi_model.pkl")
#print("\n✅ Model saved as 'random_forest_multi_model.pkl'")

# === SHAP EXPLAINABILITY ===
print("\n🔍 Starting SHAP explainability analysis...")

# Convert scaled X_test back to DataFrame with feature names
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Create directory for SHAP plots
save_dir = r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\shap"
os.makedirs(save_dir, exist_ok=True)

# Set global font to Times New Roman and increase font size
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

# Initialize TreeExplainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_df)

# Detect if multiclass
is_multiclass = isinstance(shap_values, list) and len(shap_values) > 1

# Plot global summary (multi or single)
plt.figure()
shap.summary_plot(shap_values if not is_multiclass else shap_values[0], X_test_df, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'shap_summary_plot.png'), dpi=300)
plt.show()

plt.figure()
shap.summary_plot(shap_values if not is_multiclass else shap_values[0], X_test_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'shap_bar_plot.png'), dpi=300)
plt.show()

# If multiclass, save per-class plots
if is_multiclass:
    class_labels = best_model.classes_
    for class_index, class_name in enumerate(class_labels):
        plt.figure()
        shap.summary_plot(shap_values[class_index], X_test_df, plot_type="dot", show=False)
        plt.title(f"SHAP Summary Plot - {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_summary_{class_name}.png'), dpi=300)
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values[class_index], X_test_df, plot_type="bar", show=False)
        plt.title(f"SHAP Bar Plot - {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_bar_{class_name}.png'), dpi=300)
        plt.close()

print("\n✅ SHAP explainability plots saved.")
