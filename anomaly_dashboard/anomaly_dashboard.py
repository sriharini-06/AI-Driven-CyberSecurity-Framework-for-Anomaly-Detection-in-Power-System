# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import os
import warnings
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

class DLWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        try:
            if len(self.input_shape) == 3:  # LSTM/GRU
                X_reshaped = X.reshape(self.input_shape)
            elif len(self.input_shape) == 4:  # CNN-LSTM
                # Special handling for CNN-LSTM
                if X.shape[1] != 1:  # If features need to be reshaped
                    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1], 1))
                else:
                    X_reshaped = X.reshape(self.input_shape)
            
            preds = self.model.predict(X_reshaped, verbose=0)
            return np.argmax(preds, axis=1) if len(preds.shape) > 1 and preds.shape[1] > 1 else (preds > 0.5).astype(int).flatten()
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return np.zeros(X.shape[0])

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\codes\codes\cleaned_dataset.csv")

# Load and prepare data
df = load_data()
X = df.drop(columns=["snort_alert","Use_Case"])
y = df["snort_alert"]
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Model Selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Select Model", ["Machine Learning", "Deep Learning"])
if model_choice == "Machine Learning":
    ml_model_choice = st.sidebar.selectbox("Select ML Model", 
        ["random_forest_model", "svm_model", "mlp_model", "logistic_regression_model","bnb_model","gnb_model"])
else:
    dl_model_choice = st.sidebar.selectbox("Select DL Model", 
        ["lstm_model", "cnn_lstm_model", "gru_model"])

# Model Loading and Prediction
if model_choice == "Deep Learning":
    try:
        model_path = f"C:/Users/Vignes V M/OneDrive/Desktop/clg/Semester 2/projects/EEE/codes/codes/models/{dl_model_choice}.h5"
        model = load_model(model_path)
        
        # Model-specific input shapes
        if dl_model_choice == "cnn_lstm_model":
          # CNN-LSTM expects (batch, timesteps, features, channels)
          # Verify your actual model expects 33 features with 1 channel
          input_shape = (X_test.shape[0], 1, 33, 1)  # Adjusted to match your model
          # Ensure we have exactly 33 features
          if X_test.shape[1] != 33:
            st.error(f"CNN-LSTM expects 33 features but got {X_test.shape[1]}")
            st.stop()
          X_test_adj = X_test.reshape(input_shape)
        elif dl_model_choice == "gru_model":
            input_shape = (X_test.shape[0], 1, 35)  # GRU expects 35 features
            # Pad features if needed
            if X_test.shape[1] < 35:
                X_test_adj = np.pad(X_test, ((0,0), (0,35-X_test.shape[1])))
            else:
                X_test_adj = X_test[:,:35]
        else:  # LSTM
            input_shape = (X_test.shape[0], 1, X_test.shape[1])
            X_test_adj = X_test
            
        dl_wrapper = DLWrapper(model, input_shape)
        y_pred = dl_wrapper.predict(X_test_adj)
        y_test = y_test.values
        y_pred = y_pred[:len(y_test)]
        
    except Exception as e:
        st.error(f"Error loading {dl_model_choice}: {str(e)}")
        st.stop()
else:
    model = joblib.load(f"C:/Users/Vignes V M/OneDrive/Desktop/clg/Semester 2/projects/EEE/codes/codes/models/{ml_model_choice}.pkl")
    y_pred = model.predict(X_test)

# Dashboard
st.title("🔍 AI-Driven Anomaly Detection Dashboard")

# 1. Anomaly Trends
st.subheader("📈 Anomaly Trends Over Time")
try:
    fig = px.line(pd.DataFrame({'Index': range(len(y_test)), 'Alert': y_test}), 
                 x='Index', y='Alert', title="Anomaly Detection Over Time")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error creating trend plot: {str(e)}")

# 2. Anomaly Distribution
st.subheader("📊 Anomaly Distribution")
try:
    fig, ax = plt.subplots(facecolor='none', edgecolor='white')  # Transparent background
    ax.set_facecolor('none')  # Transparent axes
    # Customize colors
    sns.countplot(x=y_pred, ax=ax, palette=['#1f77b4', '#ff7f0e'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'], color='white')  # White text
    ax.tick_params(axis='both', colors='white')  # White ticks
    ax.spines['bottom'].set_color('white')  # White axis lines
    ax.spines['left'].set_color('white')
    ax.yaxis.label.set_color('white')  # White y-axis label
    ax.xaxis.label.set_color('white')  # White x-axis label
    ax.title.set_color('white')  # White title
    st.pyplot(fig, transparent=True)
except Exception as e:
    st.error(f"Error creating distribution plot: {str(e)}")

# 3. Confusion Matrix
st.subheader("🟢 Confusion Matrix")
try:
    fig, ax = plt.subplots(facecolor='none')
    ax.set_facecolor('none')
    cm = confusion_matrix(y_test, y_pred)
    # Custom heatmap with visible text
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax,
               xticklabels=['Normal', 'Anomaly'], 
               yticklabels=['Normal', 'Anomaly'],
               annot_kws={"color": "white"})  # White annotation text
    # Set color of tick labels
    ax.set_xticklabels(ax.get_xticklabels(), color='white')
    ax.set_yticklabels(ax.get_yticklabels(), color='white')
    # Set title color
    ax.title.set_color('white')
    st.pyplot(fig, transparent=True)
except Exception as e:
    st.error(f"Error creating confusion matrix: {str(e)}")

# 4. Feature Importance
try:
    if model_choice == "Machine Learning":
        if ml_model_choice == "random_forest_model":
            st.subheader("📌 Feature Importance")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            fig, ax = plt.subplots(facecolor='none', figsize=(12, 6))
            ax.set_facecolor('none')
            plt.title("Feature Importances", color='white')
            bars = plt.bar(range(len(feature_names)), importances[indices], 
                          align="center", color="skyblue")
            plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], 
                     rotation=90, color='white')
            plt.yticks(color='white')
            plt.xlim([-1, len(feature_names)])
            # Set axis labels color
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            # Set spines color
            for spine in ax.spines.values():
                spine.set_color('white')
            st.pyplot(fig, transparent=True)
            
        elif ml_model_choice == "logistic_regression_model":
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[::-1]
            fig, ax = plt.subplots(facecolor='none', figsize=(12, 6))
            ax.set_facecolor('none')
            plt.title("Feature Coefficients (Absolute Values)", color='white')
            bars = plt.bar(range(len(feature_names)), np.abs(coef)[indices], 
                         align="center", color="salmon")
            plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], 
                     rotation=90, color='white')
            plt.yticks(color='white')
            plt.xlim([-1, len(feature_names)])
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
            st.pyplot(fig, transparent=True)
            
        elif ml_model_choice in ["svm_model", "mlp_model"]:
            result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            indices = np.argsort(result.importances_mean)[::-1]
            fig, ax = plt.subplots(facecolor='none', figsize=(12, 6))
            ax.set_facecolor('none')
            plt.title("Permutation Importance", color='white')
            bars = plt.bar(range(len(feature_names)), result.importances_mean[indices], 
                         align="center", color="lightgreen")
            plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], 
                     rotation=90, color='white')
            plt.yticks(color='white')
            plt.xlim([-1, len(feature_names)])
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
            st.pyplot(fig, transparent=True)
except Exception as e:
    st.error(f"Error creating feature importance: {str(e)}")

# 5. Performance Metrics
st.subheader("📊 Model Performance Metrics")
try:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
    with col2:
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")
        
    st.text("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())
except Exception as e:
    st.error(f"Error calculating metrics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### Model Details")
st.write(f"Currently showing results for: **{model_choice} > {ml_model_choice if model_choice == 'Machine Learning' else dl_model_choice}**")