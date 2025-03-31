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
                                               redirect_uri = 'http://127.0.0.1:8888/callback',
                                               scope = 'user-library-read'))

spotify_df = pd.read_csv('./Code/Data/dataset_new_labeled.csv')

###

class Status(Enum):
    START = 0
    GET_MOOD_USR = 0
    GET_MOOD_SONG = 1
    END = 3

current_state = Status.GET_MOOD_USR

detected_mood = None
detected_mood_music = None

RECOMMEND_SONGS = 10

fsm = 0

conversation = []

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

def get_random_song_by_mood(mood_label):
    # Filter songs with the given mood
    filtered_df = spotify_df[spotify_df['mood_label'].str.lower() == mood_label.lower()]
    
    # Check if any song matches the mood
    if filtered_df.empty:
        return f"Track Not Found."

    # Pick a random row
    song = filtered_df.sample(n=1).iloc[0]
    name_track = song['name_tracks']
    name_artist = song['name_artists']
    
    return get_spotify_link(name_track, name_artist)

def create_message(message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly assistant trained to understand the user's mood "
                "through casual and supportive conversation. "
                "Ask follow-up questions to gently confirm how the user is feeling."
            )
        },
        *conversation_history,
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content

def mood_analysis(message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    messages = [
        {
            "role": "system",
            "content": (
                "Analyze the user's message and respond with only one word "
                "indicating the detected mood: Party, Melancholic, Joyful, Sad, Motivating, Relaxed, or Undetected."
            )
        },
        *conversation_history,
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content

def create_message_music(detected_mood, message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a friendly and conversational assistant that helps users choose the kind of music they want to listen to. "
                f"The user is currently feeling '{detected_mood}'. Based on that and the conversation so far, "
                "have a natural and empathetic dialogue to discover what kind of music mood they would prefer right now. "
                "You are not classifying the music mood yourself — your goal is to guide the conversation and understand the user's preference. "
                "The available music moods are: Party, Melancholic, Joyful, Sad, Motivating or Relaxed"
                "Make the interaction feel human, supportive, and casual."
            )
        },
        *conversation_history,
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content

def recommend_music_mood(detected_mood, message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    messages = [
        {
            "role": "system",
            "content": (
                f"The user is currently feeling '{detected_mood}'. "
                "Based on this and the conversation history, determine what type of music mood they would prefer. "
                "Respond with only one with one word of the following: Party, Melancholic, Joyful, Sad, Motivating, Relaxed, or Undetected."
            )
        },
        *conversation_history,
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content.strip()

def recommend_tack(detected_mood, detected_mood_music, message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    messages = [
        {
            "role": "system",
            "content": (
                "You are a kind and empathetic assistant. "
                f"The user is currently feeling '{detected_mood}', but they prefer to listen to '{detected_mood_music}' music right now. "
                "Based on this and the conversation history, write a warm, natural, and supportive message. "
                f"You Must recommend this song {song_name} - {song_artist} with the Spotify track link ({spotify_link}) as a recommendation that matches their preferred music mood. "
                "Make it feel personal and encouraging, not robotic."
            )
        },
        *conversation_history,
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message.content.strip()

###

@app.route("/whatsapp", methods=['POST'])
def whatsapp():
    global current_state
    global detected_mood
    global detected_mood_music

    #User Whatsapp Message
    incoming_msg = request.values.get('Body', '').strip()
    #from_number = request.values.get('From', '')

    user_response = None

    if current_state == Status.GET_MOOD_USR:
        #User Interaction and User Mood Analysis
        user_response = create_message(incoming_msg, conversation)
        currentMood = mood_analysis(incoming_msg, conversation)

        if currentMood != "Undetected":
            current_state = Status.GET_MOOD_SONG
            user_response = create_message_music(detected_mood, incoming_msg, conversation)

    elif current_state == Status.GET_MOOD_SONG:
        #User Interaction and User Music Mood Analysis
        user_response = create_message_music(detected_mood, incoming_msg, conversation)
        currentMood = recommend_music_mood(detected_mood, incoming_msg, conversation)

        if currentMood != "Undetected":
            current_state = Status.END
            user_response = recommend_tack(detected_mood, detected_mood_music, incoming_msg, conversation)

    else:
        user_response = None

    if (current_state == Status.GET_MOOD_USR)  or (current_state == Status.GET_MOOD_SONG):
        conversation.append({"role": "user", "content": incoming_msg})
        conversation.append({"role": "assistant", "content": user_response})
    else:
        conversation.clear()

    #Prepare the Whatsapp response
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(user_response)

    return str(resp)

###

if __name__ == "__main__":
    app.run(debug=True, port=8888)