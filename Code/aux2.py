import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import joblib

import pandas as pd
import numpy as np

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']


df = pd.read_csv('./DataSource/tracks.csv')
df = df[FEATURES].dropna().copy()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(data_scaled, columns=FEATURES)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
scaled_df['mood'] = kmeans.fit_predict(scaled_df)

# Assuming KMeans has been applied and df['mood'] holds the cluster labels
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df[FEATURES])

# Plot with adjusted axis directions
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=scaled_df['mood'], cmap='viridis')
plt.title('Clusters in PCA Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Add grid or axes for clarity if necessary
plt.axhline(0, color='r', linewidth=2)  # Horizontal red line at y = 0
plt.axvline(0, color='r', linewidth=2)  # Vertical red line at x = 0

plt.colorbar(label='Mood')
plt.show()