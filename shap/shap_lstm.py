# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# === Step 1: Load dataset ===
data = pd.read_csv('cleaned_dataset.csv')
X = data.drop(columns=['snort_alert', 'Use_Case']).values
y = data['snort_alert'].values

# === Step 2: Scale and reshape ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# === Step 3: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, stratify=y, random_state=42)

# === Step 4: Flatten for SHAP ===
X_train_2d = X_train[:, 0, :]
X_test_2d = X_test[:, 0, :]

# === Step 5: Load trained LSTM model ===
model = load_model("models/lstm_model.h5")

# === Step 6: Define prediction function ===
def predict_fn(x):
    x_reshaped = x.reshape((x.shape[0], 1, x.shape[1]))
    return model.predict(x_reshaped)

# === Step 7: Create SHAP explainer ===
background = X_train_2d[np.random.choice(X_train_2d.shape[0], 100, replace=False)]
sample_data = X_test_2d[:100]

explainer = shap.KernelExplainer(predict_fn, background)

# === Step 8: Compute SHAP values ===
shap_values = explainer.shap_values(sample_data)

# === Step 9: Plot SHAP Summary ===
# Confirm feature names
feature_names = data.drop(columns=['snort_alert', 'Use_Case']).columns.tolist()

# Create output dir
shap_dir = r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\classification\dl\shap"
os.makedirs(shap_dir, exist_ok=True)

# Plot for each class
for i, class_shap_values in enumerate(shap_values):
    print(f"Plotting for class {i}, SHAP shape: {np.array(class_shap_values).shape}, Sample shape: {sample_data.shape}")
    plt.figure()
    shap.summary_plot(
        class_shap_values,
        sample_data,
        feature_names=feature_names,
        show=False,
        plot_size=(12, 8)
    )
    plt.title(f"SHAP Summary Plot for Class {i}", fontsize=18)
    plt.savefig(os.path.join(shap_dir, f"shap_summary_class_{i}.png"), dpi=300, bbox_inches='tight')
    plt.close()

print("✅ SHAP plots saved for all classes!")
