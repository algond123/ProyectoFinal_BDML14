import pandas as pd
import numpy as np

from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse
import spotipy
from spotipy.oauth2 import SpotifyOAuth
#from google.cloud import dialogflow_v2 as dialogflow
from flask import Flask, request

import time
import ast
import sys
import os
from enum import Enum

from access_credentials import chatgpt_api_key
from access_credentials import dialogflow_api_key
from access_credentials import spotify_client_id
from access_credentials import spotify_client_secret

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = spotify_client_id,
                                               client_secret = spotify_client_secret,
                                               redirect_uri = 'http://127.0.0.1:8888/callback',
                                               scope = 'playlist-modify-public'))

# Load your CSV data
spotify_df = pd.read_csv('./Code/Data/newdataset_labeled.csv')

###
detected_mood = None
detected_mood_music = 'Sadness/Depression'

RECOMMEND_SONGS = 10
###

def get_spotify_link(track_name, artist_name):
    query = f"track:{track_name} artist:{artist_name}"
    try:
        result = sp.search(query, type='track', limit=1)
        
        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            return track['external_urls']['spotify']
        else:
            return "Track Not Found."
    except Exception as e:
        print(f"Error fetching data for {track_name} by {artist_name}: {e}")
        return "Error"
    
def create_spotify_playlist(track_ids):

    name = 'pl_proj_mood_n1'
    user_id = sp.current_user()['id']
    playlist = sp.user_playlist_create(user=user_id, name=name, public=True)
    sp.playlist_add_items(playlist_id=playlist['id'], items=track_ids)
    return playlist['external_urls']['spotify']


def recommend_songs_by_mood(df, mood_input, list_size=10, popularity_pool=10):

    # First: safely normalize genre to lowercase for filtering
    filtered_df = df.copy()
    filtered_df['track_genre'] = filtered_df['track_genre'].astype(str).str.lower()
    filtered_df['artists'] = df['artists'].astype(str).str.lower()

    # Filter by mood and exclude latin/latino genres
    mood_df = filtered_df[
        (filtered_df['mood'] == mood_input) &
        (~filtered_df['track_genre'].isin(['latin', 'latino'])) &
        (~filtered_df['artists'].str.contains('bad bunny'))
    ]

    if mood_df.empty:
        return {"error": "No songs found for the given mood."}

    # Get the top N most popular songs (default: top 100)
    top_pool = mood_df.sort_values(by="popularity", ascending=False).head(popularity_pool)

    # Adjust if fewer songs available
    actual_size = min(list_size, len(top_pool))

    # Randomly sample from the top pool
    selected_songs = top_pool.sample(n=actual_size)

    # Create the result list
    recommendations = []
    for _, row in selected_songs.iterrows():
        recommendations.append({
            "artist": row["artists"],
            "track_name": row["track_name"],
            "popularity": row["popularity"],
            "genre": row["track_genre"],
            "spotify_link": f"https://open.spotify.com/track/{row['track_id']}",
            "track_id": row["track_id"]
        })

    return recommendations


# 1. Recommend songs
print(f"\n\nSadness/Depression\n")
songs = recommend_songs_by_mood(spotify_df, 'Sadness/Depression', 10, 100)

# 2. Print recommendations
for i, song in enumerate(songs, 1):
    print(f"{i}. {song['artist']} – {song['track_name']} (Popularity: {song['popularity']})")
    print(f"   🎧 {song['spotify_link']}")

# 3. Extract track IDs
track_ids = [song['track_id'] for song in songs]

# 4. Get user ID
user_id = sp.current_user()['id']

# 5. Create and fill playlist
playlist_url = create_spotify_playlist(track_ids)

print("🎶 Playlist created:")
print(playlist_url)


'''
# Ask for a mood
print(f"\n\nSadness/Depression\n")
songs = recommend_songs_by_mood(spotify_df, 'Sadness/Depression', 10, 100)

# Print recommendation
for i, song in enumerate(songs, 1):
    print(f"{i}. {song['artist']} – {song['track_name']} (Popularity: {song['popularity']})")
    print(f"   🎧 {song['spotify_link']}\n")

# Ask for a mood
print(f"\n\nJoy/Excitement\n")
songs = recommend_songs_by_mood(spotify_df, 'Joy/Excitement', 10, 100)

# Print recommendation
for i, song in enumerate(songs, 1):
    print(f"{i}. {song['artist']} – {song['track_name']} (Popularity: {song['popularity']})")
    print(f"   🎧 {song['spotify_link']}\n")

# Ask for a mood
print(f"\n\nAnger/Tense\n")
songs = recommend_songs_by_mood(spotify_df, 'Anger/Tense', 10, 100)

# Print recommendation
for i, song in enumerate(songs, 1):
    print(f"{i}. {song['artist']} – {song['track_name']} (Popularity: {song['popularity']})")
    print(f"   🎧 {song['spotify_link']}\n")

# Ask for a mood
print(f"\n\nCalmness/Relaxation\n")
songs = recommend_songs_by_mood(spotify_df, 'Calmness/Relaxation', 10, 100)

# Print recommendation
for i, song in enumerate(songs, 1):
    print(f"{i}. {song['artist']} – {song['track_name']} (Popularity: {song['popularity']})")
    print(f"   🎧 {song['spotify_link']}\n")'
'''