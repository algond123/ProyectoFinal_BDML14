import pandas as pd
import numpy as np
import time
import ast
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#enum here
cluster_to_mood = {
    0: 'Relaxed',
    1: 'Party',
    2: 'Melancholic',
    3: 'Joyful',
    4: 'Sad',
    5: 'Motivating'
}

# Merge Files tracks.csv and artists.csv in filtered_tracks_with_artists.csv
spotify_df = pd.read_csv('tracks.csv')
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo']
data = spotify_df[features].dropna()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)

data['mood_cluster'] = kmeans.fit_predict(data_scaled)
spotify_df['mood_cluster'] = data['mood_cluster']
spotify_df['mood_label'] = spotify_df['mood_cluster'].map(cluster_to_mood)
spotify_df.to_csv('spotify_tracks_with_moods.csv', index=False)

df_tracks = pd.read_csv('spotify_tracks_with_moods.csv')
df_tracks_cleaned = df_tracks.dropna()
df_tracks_filter = df_tracks_cleaned[['id', 'name', 'id_artists', 'mood_label']].copy()
df_tracks_filter.rename(columns={'id': 'id_tracks', 'name': 'name_tracks'}, inplace=True)
df_tracks_filter.loc[:, 'id_artists'] = df_tracks_filter['id_artists'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_tracks_filter.loc[:, 'id_artists'] = df_tracks_filter['id_artists'].apply(lambda x: x[0] if isinstance(x, list) else x)
df_tracks_filter.to_csv('filtered_tracks.csv', index=False)

df_artists = pd.read_csv('artists.csv')
df_artists_cleaned = df_artists.dropna()
df_artists_filter = df_artists_cleaned[['id', 'name']].copy()
df_artists_filter.rename(columns={'id': 'id_artists', 'name': 'name_artists'}, inplace=True)
df_artists_filter.to_csv('filtered_artists.csv', index=False)

df_artists_new = pd.read_csv('filtered_artists.csv')
df_tracks_new = pd.read_csv('filtered_tracks.csv')
merged_df = pd.merge(df_tracks_new, df_artists_new, how='left', on='id_artists')
final_df = merged_df[['id_tracks', 'name_tracks', 'id_artists', 'name_artists', 'mood_label']]
final_df = final_df[['name_tracks', 'name_artists', 'mood_label']].copy()
final_df.to_csv('filtered_tracks_with_artists.csv', index=False)

# New file with name_tracks and name_artists
df_tracks_with_artists = pd.read_csv('filtered_tracks_with_artists.csv')

# Spotify auth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="3261e217b565417eb36d76c5e369c850",
                                               client_secret="19561b0996ac4b29adaaa7afd5f90ac5",
                                               redirect_uri="http://127.0.0.1:8888/callback",
                                               scope="user-library-read"))

# Function for get spotify link
def get_spotify_link(track_name, artist_name):
    query = f"track:{track_name} artist:{artist_name}"
    try:
        result = sp.search(query, type='track', limit=1)
        
        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            return track['external_urls']['spotify']
        else:
            return "Track not found."
    except Exception as e:
        print(f"Error fetching data for {track_name} by {artist_name}: {e}")
        return "Error"
    
# Get all spotify links from the csv and save in filtered_tracks_with_spotify_links.csv
spotify_data = []

for _, row in df_tracks_with_artists.iterrows():
    track_name = row['name_tracks']
    artist_name = row['name_artists']
    
    spotify_link = get_spotify_link(track_name, artist_name)
    spotify_data.append([track_name, artist_name, spotify_link])
    
    print(f"track_name: {track_name}, artist_name: {artist_name}, spotify_link: {spotify_link}")
    time.sleep(0.001)

df_spotify_links = pd.DataFrame(spotify_data, columns=['name_tracks', 'name_artists', 'spotify_link'])
df_spotify_links.to_csv('filtered_tracks_with_spotify_links.csv', index=False)