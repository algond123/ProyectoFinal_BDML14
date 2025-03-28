import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator

import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"./DataLogged/{timestamp}_optimal_kmeans.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']



logging.info("Step 1: Loading dataset and Selecting features")
df = pd.read_csv('./DataSource/tracks.csv')
df = df[FEATURES].dropna().copy()

logging.info("Step 2: Standardizing features...")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

logging.info("Step 3: Running Kmeans from 2 to 15 Clusters")
wcss = []
sil_results = []
ch_results = []
db_results = []

k_range = range(2, 16)
for k in k_range:
    logging.info(f"  Running K-Means for k={k}...")
    start_time = time.time()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)

    wcss.append(kmeans.inertia_)
    sil_score = silhouette_score(
        data_scaled,
        kmeans.labels_,
        metric='euclidean',
        sample_size=29333,
        random_state=42,
        n_jobs=-1
    )
    sil_results.append(sil_score)
    ch_score = calinski_harabasz_score(data_scaled, labels)
    ch_results.append(ch_score)
    db_score = davies_bouldin_score(data_scaled, labels)
    db_results.append(db_score)

    logging.info(f"  Completed K={k} in {time.time() - start_time:.2f} seconds")

logging.info("Step 4: Printing Clusters with WCSS & Plotting Elbow Curve")
plt.plot(k_range, wcss, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()

logging.info("Step 5: Calculation of K-Means optimal K... with KneeLocator")
knee = KneeLocator(list(k_range), wcss, curve='convex', direction='decreasing')
logging.info(f"Optimal (knee) number of clusters (k): {knee.knee}")

logging.info("Step 6: Print Results")
results_df = pd.DataFrame({
    'k': list(k_range),
    'WCSS': wcss,
    'Silhouette Score': sil_results,
    'Calinski-Harabasz': ch_results,
    'Davies-Bouldin': db_results
})

logging.info(f"\n{results_df.round(5).to_string(index=False)}")

logging.info("All steps completed!")