import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import pandas as pd
import numpy as np

# Optimal number of clusters
k_optimal = 4

# Features to be used in the analysis
#FEATURES = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
#FEATURES = ['valence','energy','tempo','loudness','danceability','acousticness','instrumentalness','speechiness','liveness']

FEATURES = ['valence','energy','danceability','tempo','loudness']
#FEATURES = ['valence','energy','danceability','tempo','loudness', 'liveness','speechiness']
#FEATURES = ['valence','energy','danceability','tempo','loudness', 'liveness','speechiness','acousticness','instrumentalness']

# Load the dataset
df = pd.read_csv('./DataSource/tracks.csv')
df = df[FEATURES].dropna().copy()

# Standardize the data (scaling to mean=0, std=1)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Perform KMeans clustering
scaled_df = pd.DataFrame(data_scaled, columns=FEATURES)
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
scaled_df['mood'] = kmeans.fit_predict(scaled_df)

# Perform PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df[FEATURES])

# 1. Plotting the heatmap of the clusters first (based on mood)
plt.figure(figsize=(10, 7))
sns.heatmap(scaled_df.groupby('mood').mean().T, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Cluster Centers')
plt.ylabel('Features')
plt.xlabel('Cluster')
plt.show()

# 2. Plotting the clusters in PCA space (after the heatmap)
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=scaled_df['mood'], cmap='viridis')
plt.title('Clusters in PCA Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.axhline(0, color='r', linewidth=2)
plt.axvline(0, color='r', linewidth=2)

plt.colorbar(label='Mood')
plt.show()

# Create a DataFrame to analyze the correlation between PCA components and original features
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df = pd.concat([pca_df, scaled_df], axis=1)
correlation_matrix = pca_df.corr()

# 3. Plotting the heatmap of the correlation between PCA components and features
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation between PCA Components and Features")
plt.show()

# 4. Plotting the PCA biplot (after the correlation heatmap)
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', alpha=0.5, label="Songs")

for i, feature in enumerate(FEATURES):
    plt.quiver(0, 0, pca.components_[0, i], pca.components_[1, i], angles='xy', scale_units='xy', scale=1, label=feature)

plt.title('PCA Biplot')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()