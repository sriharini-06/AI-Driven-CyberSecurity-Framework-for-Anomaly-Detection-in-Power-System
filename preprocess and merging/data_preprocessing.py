# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Enable UTF-8 encoding

# Load dataset
file_path = "merged_dataset.csv" 
df = pd.read_csv(file_path)

# Fix: Assign the results explicitly instead of using `inplace=True`
df["DNP3 Objects"] = df["DNP3 Objects"].fillna(df["DNP3 Objects"].median())
df["value1"] = df["value1"].fillna(df["value1"].mean())
df["value2"] = df["value2"].fillna(df["value2"].mean())
df["value3"] = df["value3"].fillna(df["value3"].mean())
df["value4"] = df["value4"].fillna(df["value4"].mean())

# Drop 'Time' column (Not useful for model training)
df = df.drop(columns=["Time","Unnamed: 0"])

# Encode 'Use_Case' (Convert categorical values into numbers)
df["Use_Case"] = df["Use_Case"].astype("category").cat.codes

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("✅ Data cleaning complete! Saved as 'cleaned_dataset.csv'.")
