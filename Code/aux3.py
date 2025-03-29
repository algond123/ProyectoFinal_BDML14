import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Features for different experiments
FEATURES_1 = ['valence', 'energy', 'danceability', 'tempo', 'loudness']
FEATURES_2 = ['valence', 'energy', 'danceability', 'tempo', 'loudness', 'liveness']
FEATURES_3 = ['valence', 'energy', 'danceability', 'tempo', 'loudness', 'speechiness']
FEATURES_4 = ['valence', 'energy', 'danceability', 'tempo', 'loudness', 'liveness', 'speechiness']
FEATURES_5 = ['valence', 'energy', 'danceability', 'tempo', 'loudness', 'liveness', 'speechiness', 'acousticness', 'instrumentalness']

# Function to plot the clusters in PCA space
def plot_pca_clusters_all(scaled_df, kmeans, FEATURES, ax, label):
    # Apply PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_df[FEATURES])

    # Plot clusters in PCA space on the same axes
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=scaled_df['mood'], cmap='viridis', alpha=0.5)
    ax.set_title(f'Clusters in PCA Space ({label})')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

    # Adding grid and axis lines for clarity
    ax.axhline(0, color='r', linewidth=2)  # Horizontal red line at y = 0
    ax.axvline(0, color='r', linewidth=2)  # Vertical red line at x = 0

    return scatter

# Load your data
df = pd.read_csv('./DataSource/tracks.csv')

# Create a single figure for all plots
fig, axes = plt.subplots(5, 4, figsize=(20, 20))
fig.subplots_adjust(hspace=0.4)

# Flatten axes for easier indexing
axes = axes.flatten()

# Loop over each set of features and each k value (3 to 6)
counter = 0
for FEATURES in [FEATURES_1, FEATURES_2, FEATURES_3, FEATURES_4, FEATURES_5]:
    df_filtered = df[FEATURES].dropna().copy()  # Drop rows with missing values
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_filtered)
    
    # Convert the scaled data back to DataFrame for clustering
    scaled_df = pd.DataFrame(data_scaled, columns=FEATURES)
    
    # Loop over different values of k (3 to 6)
    for k in range(3, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        scaled_df['mood'] = kmeans.fit_predict(scaled_df)
        
        # Plot the clusters in PCA space on the same axes
        scatter = plot_pca_clusters_all(scaled_df, kmeans, FEATURES, axes[counter], f'k={k}, Features={len(FEATURES)}')
        
        # Add colorbar for each subplot
        fig.colorbar(scatter, ax=axes[counter])
        
        counter += 1

# Show the plot
plt.show()
