# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import pandas as pd
import numpy as np
import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ----------------------------
# Function: Create sequences
# ----------------------------
def create_sequences(X, y, seq_length):
    sequences, labels = [], []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        labels.append(y[i+seq_length])
    return np.array(sequences), np.array(labels)

# ----------------------------
# PGD Attack Function
# ----------------------------
def pgd_attack(model, X, y, epsilon=0.01, alpha=0.002, num_iter=10):
    X_adv = tf.convert_to_tensor(X, dtype=tf.float32)

    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(X_adv)
            predictions = model(X_adv, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)

        # gradient
        gradient = tape.gradient(loss, X_adv)
        signed_grad = tf.sign(gradient)

        # perturbation step
        X_adv = X_adv + alpha * signed_grad

        # clip to epsilon-ball
        X_adv = tf.clip_by_value(X_adv, X - epsilon, X + epsilon)

    return X_adv.numpy()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained LSTM model")
    parser.add_argument("--data_path", type=str, default="cleaned_dataset.csv", help="Path to cleaned dataset")
    parser.add_argument("--seq_length", type=int, default=50, help="Sequence length used in training")
    parser.add_argument("--results_out", type=str, default="pgd_results.csv", help="Path to save results CSV")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Max perturbation")
    parser.add_argument("--alpha", type=float, default=0.002, help="Step size for PGD")
    parser.add_argument("--pgd_steps", type=int, default=10, help="Number of PGD iterations")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.data_path)

    # Separate features and labels
    X = df.drop(columns=["Use_Case"]).values
    y = df["Use_Case"].values

    # Create sequences
    X_seq, y_seq = create_sequences(X, y, args.seq_length)

    print(f"✅ Dataset loaded. Shape: {X_seq.shape}, Labels: {len(y_seq)}")

    # Load trained model
    model = load_model(args.model_path)
    print(f"✅ Model loaded from {args.model_path}")

    # Evaluate on clean data
    y_pred_clean = np.argmax(model.predict(X_seq), axis=1)
    acc_clean = accuracy_score(y_seq, y_pred_clean)
    f1_clean = f1_score(y_seq, y_pred_clean, average="weighted")

    # Run PGD attack
    X_adv = pgd_attack(model, X_seq, y_seq, epsilon=args.epsilon, alpha=args.alpha, num_iter=args.pgd_steps)

    # Evaluate on adversarial data
    y_pred_adv = np.argmax(model.predict(X_adv), axis=1)
    acc_adv = accuracy_score(y_seq, y_pred_adv)
    f1_adv = f1_score(y_seq, y_pred_adv, average="weighted")

    # Save results
    results = pd.DataFrame([{
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "pgd_steps": args.pgd_steps,
        "acc_clean": acc_clean,
        "f1_clean": f1_clean,
        "acc_adv": acc_adv,
        "f1_adv": f1_adv
    }])
    results.to_csv(args.results_out, index=False)

    print("✅ Results saved at", args.results_out)
    print(results)