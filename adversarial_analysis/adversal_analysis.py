# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
# Step 1: Load the updated model
model = load_model("models/lstm_model_adversarial.h5")
print("Adversarially trained model loaded successfully!")

# Step 2: Load and preprocess dataset
data = pd.read_csv('cleaned_dataset.csv')
X = data.drop(columns=['snort_alert', 'Use_Case']).values
y = data['snort_alert'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape((X.shape[0], 1, X.shape[1]))

num_classes = len(np.unique(y))
y_categorical = to_categorical(y, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y)

# Step 3: Generate adversarial samples for evaluation
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

epsilon = 0.02
X_test_adv = generate_adversarial_samples(model, X_test, y_test, epsilon=epsilon)

# Step 4: Evaluate on adversarial samples
y_pred_adv = model.predict(X_test_adv)
y_pred_adv_classes = np.argmax(y_pred_adv, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

adversarial_accuracy = np.mean(y_true_classes == y_pred_adv_classes)
print(f"\nAdversarial Accuracy (epsilon={epsilon}): {adversarial_accuracy:.4f}")

confusion_adv = confusion_matrix(y_true_classes, y_pred_adv_classes)
print("\nConfusion Matrix on Adversarial Samples:")
print(confusion_adv)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_adv, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],annot_kws={"size": 40, "fontname":"Times New Roman"},cbar=False)
plt.xlabel('Predicted', fontsize=40, fontname="Times New Roman")
plt.ylabel('True', fontsize=40, fontname="Times New Roman")
plt.xticks(fontsize=30, fontname="Times New Roman")
plt.yticks(fontsize=30, fontname="Times New Roman")
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\classification", "adversarial.png"), dpi=300)       
plt.show()
