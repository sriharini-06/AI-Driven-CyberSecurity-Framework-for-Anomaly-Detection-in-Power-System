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
import tensorflow as tf

# Step 1: Load the dataset
data = pd.read_csv('cleaned_dataset.csv')
X = data.drop(columns=['snort_alert', 'Use_Case']).values  # Features
y = data['snort_alert'].values                             # Labels

# Step 2: Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Balance the training data
X_train_2d = X_train.reshape(X_train.shape[0], -1)
smote = SMOTE(random_state=42)
X_train_balanced_2d, y_train_balanced = smote.fit_resample(X_train_2d, y_train)
X_train_balanced = X_train_balanced_2d.reshape(X_train_balanced_2d.shape[0], 1, X_train_balanced_2d.shape[1])

# Step 5: Convert labels to categorical
num_classes = len(np.unique(y))
y_train_balanced_categorical = to_categorical(y_train_balanced, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Step 6: Define FGSM Adversarial Attack Function
def generate_adversarial_samples(model, X, y, epsilon=0.02):
    X_adv = tf.convert_to_tensor(X)
    y_true = tf.convert_to_tensor(y)
    with tf.GradientTape() as tape:
        tape.watch(X_adv)
        predictions = model(X_adv)
        loss = tf.keras.losses.categorical_crossentropy(y_true, predictions)
    gradient = tape.gradient(loss, X_adv)
    perturbation = epsilon * tf.sign(gradient)
    X_adv = X_adv + perturbation
    X_adv = tf.clip_by_value(X_adv, clip_value_min=-1, clip_value_max=1)
    return X_adv.numpy()

# Step 7: Build the LSTM model
model = Sequential([
    LSTM(128, input_shape=(X_train_balanced.shape[1], X_train_balanced.shape[2]), activation='tanh'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: Generate adversarial samples for training
X_train_adv = generate_adversarial_samples(model, X_train_balanced, y_train_balanced_categorical, epsilon=0.02)

# Combine clean and adversarial data
X_train_combined = np.concatenate([X_train_balanced, X_train_adv], axis=0)
y_train_combined = np.concatenate([y_train_balanced_categorical, y_train_balanced_categorical], axis=0)

# Step 9: Train the model with combined data
history = model.fit(
    X_train_combined,
    y_train_combined,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

accuracy = np.mean(y_test == y_pred_classes)
print(f"\nAccuracy: {accuracy:.6f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))


# Save the updated model
model.save("models/lstm_model_adversarial.h5")
