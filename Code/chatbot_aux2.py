import pandas as pd
import numpy as np

from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse
import spotipy
from spotipy.oauth2 import SpotifyOAuth
#from google.cloud import dialogflow_v2 as dialogflow

from flask import Flask, request
import logging
import time
import ast
import sys
import os
from enum import Enum

from access_credentials import chatgpt_api_key
from access_credentials import dialogflow_api_key
from access_credentials import spotify_client_id
from access_credentials import spotify_client_secret

#logging.basicConfig(level=logging.DEBUG)

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = spotify_client_id,
                                               client_secret = spotify_client_secret,
                                               redirect_uri = 'http://127.0.0.1:8888/spotify',
                                               scope = 'playlist-modify-public'))

# Load your CSV data
spotify_df = pd.read_csv('./Code/Data/newdataset_labeled.csv')

###
detected_mood = None
detected_mood_music = None

RECOMMEND_SONGS = 10
###

def get_spotify_track_info(track_name, artist_name):
    query = f"track:{track_name} artist:{artist_name}"
    try:
        result = sp.search(query, type='track', limit=1)
        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            return {
                "track_id": track["id"],
                "spotify_link": track["external_urls"]["spotify"]
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching data for {track_name} by {artist_name}: {e}")
        return None

def recommend_songs_by_mood(df, mood_input, list_size=10, popularity_pool=30):
    filtered_df = df.copy()
    filtered_df['track_genre'] = filtered_df['track_genre'].astype(str).str.lower()
    filtered_df['artists'] = df['artists'].astype(str).str.lower()

    mood_df = filtered_df[
        (filtered_df['mood'] == mood_input) &
        (~filtered_df['track_genre'].isin(['latin', 'latino'])) &
        (~filtered_df['artists'].str.contains('bad bunny'))
    ]

    if mood_df.empty:
        return {"error": "No songs found for the given mood."}

    top_pool = mood_df.sort_values(by="popularity", ascending=False).head(popularity_pool)
    selected_songs = top_pool.sample(n=min(list_size * 2, len(top_pool)))

    recommendations = []
    for _, row in selected_songs.iterrows():
        artist = row["artists"]
        track_name = row["track_name"]
        track_info = get_spotify_track_info(track_name, artist)

        if track_info:
            recommendations.append({
                "artist": artist,
                "track_name": track_name,
                "popularity": row["popularity"],
                "genre": row["track_genre"],
                "track_id": track_info["track_id"],
                "spotify_link": track_info["spotify_link"]
            })

        if len(recommendations) >= list_size:
            break

    if not recommendations:
        return {"error": "No valid Spotify results found."}

    return recommendations

def create_spotify_playlist(track_ids, mood_input):

    name = 'pl_proj_mood_' + mood_input
    user_id = sp.current_user()['id']
    playlist = sp.user_playlist_create(user=user_id, name=name, public=True)
    sp.playlist_add_items(playlist_id=playlist['id'], items=track_ids)
    return playlist['external_urls']['spotify']

detected_mood_music = 'Sadness/Depression'

# 1. Recommend songs
print(f"{detected_mood_music}")
songs = recommend_songs_by_mood(spotify_df, detected_mood_music, 10, 100)

'''
# 2. Print recommendation
for i, song in enumerate(songs, 1):
    print(f"{i}. {song['artist']} – {song['track_name']} (Popularity: {song['popularity']})")
    print(f"   🎧 {song['spotify_link']}")
'''

# 3. Extract track IDs
track_ids = [song['track_id'] for song in songs]

# 4. Create and fill playlist
playlist_url = create_spotify_playlist(track_ids, detected_mood_music)

print("🎶 Playlist created:")
print(playlist_url)