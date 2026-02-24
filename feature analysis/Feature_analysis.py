# Vignes V M (24359) - Sri Harini M P (24358) - Rohith Ravi (24350) - Yashwanth B (24360)
from enum import unique
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
# Load the dataset
data = pd.read_csv("cleaned_dataset.csv")

# Prepare features and target
X = data.drop(['snort_alert',"snort_alert_type"], axis=1)
y = data['snort_alert']
# ----- STEP 1: Data Distribution Analysis -----
print("Performing Data Distribution Analysis...")
for column in X.columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=X, x=column, hue=y, fill=True, palette='bright')
    plt.xlabel(column,fontsize=40, fontname="Times New Roman")
    plt.ylabel('Density',fontsize=40, fontname="Times New Roman")
    plt.xticks(fontsize=30, fontname="Times New Roman")                                                                                                                     
    plt.yticks(fontsize=30,fontname="Times New Roman")
    plt.legend(title="Class", labels=y.unique(),fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\Feature_Analysis\data_dis", f"{column}.png"), dpi=300)
    plt.close()

# ----- STEP 2: Correlation Analysis -----
print("Performing Correlation Analysis...")
# Select only numeric columns
numeric_data = data

# Compute the correlation matrix for numeric columns
correlation_matrix = numeric_data.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.xticks(fontsize=20, fontname="Times New Roman")
plt.yticks(fontsize=20, fontname="Times New Roman")
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\Feature_Analysis", "correlation_heatmap.png"), dpi=300)
plt.close()

# ----- STEP 3: Feature Importance using Random Forest -----
print("Calculating Feature Importance with Random Forest...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
importances = rf_model.feature_importances_
feature_names = X.columns
sorted_indices = importances.argsort()

# Generate colors using a colormap (e.g., 'viridis')
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(sorted_indices)))

# Plot feature importance (Random Forest)
plt.figure(figsize=(12, 8))
plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center', color=colors)
plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
plt.xlabel('Importance Score',fontsize=40, fontname="Times New Roman")
plt.ylabel('Features',fontsize=40, fontname="Times New Roman")
plt.yticks(fontsize=15, fontname="Times New Roman")
plt.xticks(fontsize=15, fontname="Times New Roman")
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\Feature_Analysis", "Random_forest_fa.png"), dpi=300)
plt.show()

# ----- STEP 5: Dimensionality Reduction using PCA -----
print("Performing Dimensionality Reduction using PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='bright', s=20)
plt.xlabel('Principal Component 1',fontsize=35, fontname="Times New Roman")
plt.ylabel('Principal Component 2',fontsize=35, fontname="Times New Roman")
plt.xticks(fontsize=20, fontname="Times New Roman")
plt.yticks(fontsize=20, fontname="Times New Roman")
plt.legend(title='Class', loc='upper right',fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(r"C:\Users\Vignes V M\OneDrive\Desktop\clg\Semester 2\projects\EEE\graphs\Feature_Analysis", "pca_2d.png"), dpi=300,transparent=True)
plt.show()

print("Feature Analysis Complete.")
