# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')  # Enable UTF-8 encoding

# Load the dataset
file_path = "merged_dataset.csv"  # Update if needed
df = pd.read_csv(file_path)

# Display basic info
print("\n🔹 Dataset Info:")
print(df.info())

# Display the first few rows
print("\n🔹 First 5 Rows:")
print(df.head())

# Check for missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\n🔹 Duplicates:")
print(f"Total Duplicate Rows: {df.duplicated().sum()}")

# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Select only numerical columns for correlation matrix
numerical_df = df.select_dtypes(include=["int64", "float64"])

# Generate heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_df.corr(), annot=False, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Detect categorical and numerical features
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

print(f"\n🔹 Categorical Columns: {list(categorical_cols)}")
print(f"🔹 Numerical Columns: {list(numerical_cols)}")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["font.weight"] = "bold"

# Plot distribution of numerical features
df[numerical_cols].hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\Feature_Analysis", "numerical_distributions.png"), dpi=300)
plt.show()
