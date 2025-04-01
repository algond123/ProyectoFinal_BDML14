import os
import sys
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
import logging
from enum import Enum

from access_credentials import chatgpt_api_key
#from access_credentials import dialogflow_api_key
from access_credentials import spotify_client_id
from access_credentials import spotify_client_secret

#logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


#The trained is not working ok on Dialogflow
'''
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = dialogflow_api_key
session_client = dialogflow.SessionsClient()
session = session_client.session_path('keepcoding-436818', 'duver321')
def detect_intent_texts(text):
    text_input = dialogflow.TextInput(text=text, language_code='en')
    query_input = dialogflow.QueryInput(text=text_input)

    response = session_client.detect_intent(request={'session': session, 'query_input': query_input})

    return response.query_result.fulfillment_text
'''

###

client = OpenAI(
  api_key = chatgpt_api_key
  #api_key="xxx"
)

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = spotify_client_id,
                                               client_secret = spotify_client_secret,
                                               redirect_uri = 'http://127.0.0.1:8888/spotify',
                                               scope = 'playlist-modify-public'))

spotify_df = pd.read_csv('./Data/dataset_new_labeled.csv')

###

mood_keywords = {
    'sadness': 'Sadness/Depression',
    'depression': 'Sadness/Depression',
    'joy': 'Joy/Excitement',
    'excitement': 'Joy/Excitement',
    'anger': 'Anger/Tense',
    'tense': 'Anger/Tense',
    'calmness': 'Calmness/Relaxation',
    'relaxation': 'Calmness/Relaxation',
}

PLAYLIST_SIZE = 10
TOP_POPULARITY = 100

class Prompt(Enum):
    USR_CHAT_MOOD = 0
    USR_MOOD_ANALYSIS = 1
    USR_CHAT_MUSIC_MOOD = 2
    USR_MUSIC_MOOD_ANALYSIS = 3
    USR_CHAT_PLAYLIST_RECOMMENDATION = 4

class Status(Enum):
    GET_MOOD_USR = 0
    GET_MOOD_SONG = 1

current_state = Status.GET_MOOD_USR
detected_mood_user = None
detected_mood_music = None
conversation = []

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

def create_spotify_playlist(track_ids):

    name = 'Melody Miners Recommendations'
    user_id = sp.current_user()['id']
    playlist = sp.user_playlist_create(user=user_id, name=name, public=True)
    sp.playlist_add_items(playlist_id=playlist['id'], items=track_ids)
    return playlist['external_urls']['spotify']

###

def prompt_creation(prompt_enum, mood_user : str = None, mood_music : str = None, playlist_link : str = None):

    prompt_ret = None

    if prompt_enum == Prompt.USR_CHAT_MOOD:
        prompt_ret = (
            f"You are a friendly assistant trained to understand the user's mood. "
            f"through casual and supportive conversation. "
            f"Ask follow-up questions to gently confirm how the user is feeling. "
        )
    elif prompt_enum == Prompt.USR_MOOD_ANALYSIS:
        prompt_ret = (
            f"Analyze the user's message and respond with only one word. "
            f"indicating the detected mood: Sadness, Depression, Joy, Excitement, Anger, Tense, Calmness, Relaxation, or Undetected. "
        )
    elif prompt_enum == Prompt.USR_CHAT_MUSIC_MOOD:
        prompt_ret = (
            f"You are a friendly and conversational assistant that helps users choose the kind of music they want to listen to. "
            f"The user is currently feeling '{mood_user}'. Based on that and the conversation so far, "
            f"have a natural and empathetic dialogue to discover what kind of music mood they would prefer right now. "
            f"You are not classifying the music mood yourself — your goal is to guide the conversation and understand the user's preference. "
            f"The available music moods are: Sadness, Depression, Joy, Excitement, Anger, Tense, Calmness or Relaxation. "
            f"Make the interaction feel human, supportive, and casual.  but focus on understand music they want to listen "
        )
    elif prompt_enum == Prompt.USR_MUSIC_MOOD_ANALYSIS:
        prompt_ret = (
            f"The user is currently feeling '{mood_user}'. "
            f"Based on this and the conversation history, determine what type of music mood they would prefer. Respond with only one word. "
            f"Respond with only one word from the following: Sadness, Depression, Joy, Excitement, Anger, Tense, Calmness, Relaxation, or Undetected. "
        )
    elif prompt_enum == Prompt.USR_CHAT_PLAYLIST_RECOMMENDATION:
        prompt_ret = (
            f"You are a kind and empathetic assistant. "
            f"The user is currently feeling '{mood_user}', but they prefer to listen to '{mood_music}' music right now. "
            f"Based on this and the conversation history, write a warm, natural, and supportive message. "
            f"You Must recommend this Spotify playlist ({playlist_link}) as a recommendation that matches their preferred music mood. "
            f"Make it feel personal and encouraging, not robotic. "
        )
    else:
        prompt_ret = f"You are a helpful assistant. "

    return prompt_ret

def gpt_chatbot_response(prompt_enum, user_message, conversation_history = None, mood_user : str = None, mood_music : str = None, playlist_link : str = None):

    if conversation_history is None:
        conversation_history = []

    if not user_message:
        return "I didn't catch that. Could you say it another way?"

    system_prompt = prompt_creation(prompt_enum, mood_user, mood_music, playlist_link)

    if not system_prompt:
        system_prompt = "You are a helpful assistant."

    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.65
    )

    return response.choices[0].message.content

def save_conversation(user_msg, bot_msg, conversation_history):
    conversation_history.append({"role": "user", "content": user_msg})
    conversation_history.append({"role": "assistant", "content": bot_msg})

###

@app.route("/whatsapp", methods=['POST'])
def whatsapp():
    global current_state
    global detected_mood_user
    global detected_mood_music

    #User Whatsapp Message
    user_msg = request.values.get('Body', '').strip()
    #user_number = request.values.get('From', '')

    bot_msg = None
 
    if current_state == Status.GET_MOOD_USR:
        #User Interaction and User Mood Analysis
        bot_msg = gpt_chatbot_response(Prompt.USR_CHAT_MOOD, user_msg, conversation)
        current_mood = gpt_chatbot_response(Prompt.USR_MOOD_ANALYSIS, user_msg, conversation)

        if current_mood:
            current_mood = current_mood.strip().lower()

            if current_mood != "undetected":
                current_state = Status.GET_MOOD_SONG
                detected_mood_user = current_mood
                print(f"Detected User Mood: {detected_mood_user}")

                bot_msg = gpt_chatbot_response(Prompt.USR_CHAT_MUSIC_MOOD, user_msg, conversation, detected_mood_user)
        
        save_conversation(user_msg, bot_msg, conversation)

    elif current_state == Status.GET_MOOD_SONG:
        #User Interaction and User Music Mood Analysis
        bot_msg = gpt_chatbot_response(Prompt.USR_CHAT_MUSIC_MOOD, user_msg, conversation, detected_mood_user)
        current_mood = gpt_chatbot_response(Prompt.USR_MUSIC_MOOD_ANALYSIS, user_msg, conversation, detected_mood_user)

        if current_mood:
            current_mood = current_mood.strip().lower()
            
            if current_mood != "undetected":
                current_state = Status.GET_MOOD_USR
                detected_mood_music = current_mood
                print(f"Detected Music Mood: {detected_mood_music}")

                conv_music_mood = mood_keywords.get(detected_mood_music, None)
                songs = recommend_songs_by_mood(spotify_df, conv_music_mood, PLAYLIST_SIZE, TOP_POPULARITY)
                track_ids = [song['track_id'] for song in songs]
                playlist_url = create_spotify_playlist(track_ids)
                print(f"Playlist Url: {playlist_url}")

                bot_msg = gpt_chatbot_response(Prompt.USR_CHAT_PLAYLIST_RECOMMENDATION, user_msg, conversation, detected_mood_user, detected_mood_music, playlist_url)

                detected_mood_user = None
                detected_mood_music = None
                conversation.clear()

        save_conversation(user_msg, bot_msg, conversation)
    else:
        bot_msg = "Sorry, something went wrong. Let's start again."

    #Prepare the Whatsapp response
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(bot_msg)

    return str(resp)

###

if __name__ == "__main__":
    app.run(debug=True, port=8888)