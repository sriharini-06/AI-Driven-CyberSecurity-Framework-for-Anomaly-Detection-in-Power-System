# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Step 1: Load the dataset
# Replace 'your_dataset.csv' with your dataset path
data = pd.read_csv('cleaned_dataset.csv')

# Replace 'target_column' with the name of the target column in your dataset
X = data.drop(columns=['snort_alert','Use_Case']).values  # Features
y = data['snort_alert'].values                 # Labels

# Step 2: Preprocess the data
# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape X for LSTM (samples, timesteps, features)
# Assume each sample corresponds to a sequence of length 1 (timesteps=1)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Balance the training dataset using SMOTE
# Reshape X_train to 2D for SMOTE (SMOTE does not support 3D inputs)
X_train_2d = X_train.reshape(X_train.shape[0], -1)
smote = SMOTE(random_state=42)
X_train_balanced_2d, y_train_balanced = smote.fit_resample(X_train_2d, y_train)

# Reshape back to 3D for LSTM
X_train_balanced = X_train_balanced_2d.reshape(X_train_balanced_2d.shape[0], 1, X_train_balanced_2d.shape[1])

print(f"Original training set size: {X_train.shape[0]}, Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Balanced training set size: {X_train_balanced.shape[0]}, Class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# Step 5: Convert labels to categorical (for multiclass classification)
num_classes = len(np.unique(y))
y_train_balanced_categorical = to_categorical(y_train_balanced, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Step 6: Build the LSTM model
model = Sequential([
    LSTM(128, input_shape=(X_train_balanced.shape[1], X_train_balanced.shape[2]), activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the LSTM model
history = model.fit(
    X_train_balanced,
    y_train_balanced_categorical,
    epochs=17,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Step 8: Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

accuracy = np.mean(y_test == y_pred_classes)
print(f"\nAccuracy: {accuracy:.6f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

rf_cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],annot_kws={"size": 30, "fontname": "Times New Roman"},cbar=False)  
plt.xlabel('Predicted',fontsize=40,fontname="Times New Roman")
plt.ylabel('True',fontsize=40,fontname="Times New Roman")
plt.xticks(fontsize=30,fontname="Times New Roman")
plt.yticks(fontsize=30,fontname="Times New Roman")
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\classification\dl",'lstm.png'),dpi=300)
plt.show()

#save model
model.save("models/lstm_model.h5")