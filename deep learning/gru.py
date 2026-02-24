# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Split features and target
X = df.drop(columns=["snort_alert"])
y = df["snort_alert"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for GRU
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to balance training data
# Reshape X_train to 2D for SMOTE
X_train_2d = X_train.reshape(X_train.shape[0], -1)
smote = SMOTE(random_state=42)
X_train_balanced_2d, y_train_balanced = smote.fit_resample(X_train_2d, y_train)

# Reshape back to 3D for GRU
X_train_balanced = X_train_balanced_2d.reshape(X_train_balanced_2d.shape[0], 1, X_train_balanced_2d.shape[1])

# Print data distribution
print(f"Original training set size: {X_train.shape[0]}, Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Balanced training set size: {X_train_balanced.shape[0]}, Class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# Define GRU Model
model = Sequential([
    Bidirectional(GRU(256, return_sequences=True, activation="tanh", recurrent_dropout=0.2), input_shape=(X_train_balanced.shape[1], X_train_balanced.shape[2])),
    BatchNormalization(),
    Bidirectional(GRU(128, return_sequences=True, activation="tanh", recurrent_dropout=0.2)),
    BatchNormalization(),
    GRU(64, activation="tanh", recurrent_dropout=0.2),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="binary_crossentropy", metrics=["accuracy"])

# Train Model with Optimized Callbacks
model.fit(
    X_train_balanced, 
    y_train_balanced, 
    epochs=5,
    batch_size=64, 
    validation_data=(X_test, y_test), 
    verbose=2, 
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),  # Faster early stopping
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)  # Quicker learning rate reduction
    ]
)


# Evaluate Model
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n🔹 GRU Classification Report:\n", classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔹 GRU Accuracy: {accuracy:.6f}")

rf_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],annot_kws={"size": 30, "fontname": "Times New Roman"},cbar=False)
plt.xlabel('Predicted',fontsize=40,fontname="Times New Roman")
plt.ylabel('True',fontsize=40,fontname="Times New Roman")
plt.xticks(fontsize=30,fontname="Times New Roman")
plt.yticks(fontsize=30,fontname="Times New Roman")
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\classification\dl",'gru.png'),dpi=300)
plt.show()

# Save Model
model.save("models/gru_model_with_smote.h5")
