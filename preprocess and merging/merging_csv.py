# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
import pandas as pd
import os

# Define dataset path
csv_folder = r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\dataset\Datasets\csvs"

# List of CSV files to merge
csv_files = [
    "UC1/DS_merged_phy_cyb_10os_30poll_encoded.csv",
    "UC1/DS_merged_phy_cyb_10os_60poll_encoded.csv",
    "UC2/uc2_DS_merged_phy_cyb_5os_30poll_encoded.csv",
    "UC2/uc2_DS_merged_phy_cyb_5os_60poll_encoded.csv",
    "UC2/uc2_DS_merged_phy_cyb_10os_30poll_encoded.csv",
    "UC2/uc2_DS_merged_phy_cyb_10os_60poll_encoded.csv",
    "UC3/uc3_DS_merged_phy_cyb_5os_30poll_encoded.csv",
    "UC3/uc3_DS_merged_phy_cyb_5os_60poll_encoded.csv",
    "UC3/uc3_DS_merged_phy_cyb_10os_30poll_encoded.csv",
    "UC3/uc3_DS_merged_phy_cyb_10os_60poll_encoded.csv",
    "UC4/uc4_DS_merged_phy_cyb_5os_30poll_encoded.csv",
    "UC4/uc4_DS_merged_phy_cyb_5os_60poll_encoded.csv",
    "UC4/uc4_DS_merged_phy_cyb_10os_30poll_encoded.csv",
    "UC4/uc4_DS_merged_phy_cyb_10os_60poll_encoded.csv",
]

# Initialize an empty DataFrame
merged_df = pd.DataFrame()

# Loop through and merge all CSVs
for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    
    if os.path.exists(file_path):
        temp_df = pd.read_csv(file_path)
        
        # Add "Use_Case" column to track which dataset the row came from
        temp_df["Use_Case"] = file.split("/")[0]  # Extract UC1, UC2, etc.
        
        # Append to merged DataFrame
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)

# Save merged dataset
merged_csv_path = r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\codes\codes\merged_dataset.csv"
merged_df.to_csv(merged_csv_path, index=False)
print(f"Merged dataset saved as {merged_csv_path}")
