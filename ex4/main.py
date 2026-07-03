import pandas as p
import matplotlib.pyplot as pl
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Load the Iris dataset
i= load_iris()
#dataframe
data = p.DataFrame(i.data, columns=i.feature_names)
data['Species'] = i.target
print("\nFirst 5 Records:")
print(data.head())
#Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.iloc[:, :-1])  # excluding target column
print("\n Standardized Data (first 5 rows):")
print(X_scaled[:5])
# Apply PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
#  Display transformed dataset
pca_df = p.DataFrame(X_pca, columns=['PC1', 'PC2'])
print("\n PCA Transformed Data (first 5 rows):")
print(pca_df.head())
#  variance ratio
print("\n Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
# Total variance retained
total_variance = sum(pca.explained_variance_ratio_)
print("\nTotal Variance:", total_variance)
# Compare dataset shape
print("\nShape Comparison:")
print("Original Shape:", data.iloc[:, :-1].shape)
print("After PCA Shape:", X_pca.shape)
#scatter plot
pl.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Species'])
pl.xlabel("Principal Component 1")
pl.ylabel("Principal Component 2")
pl.title("PCA on Iris Dataset")
pl.show()