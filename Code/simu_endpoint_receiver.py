
from flask import Flask, request, jsonify
import sqlite3
import os
import json
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

DB_PATH = './Code/Data/spotify_data.db'

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT UNIQUE,
                artists TEXT,
                album_name TEXT,
                track_name TEXT,
                popularity INTEGER,
                duration_ms INTEGER,
                explicit BOOLEAN,
                danceability REAL,
                energy REAL,
                key INTEGER,
                loudness REAL,
                mode INTEGER,
                speechiness REAL,
                acousticness REAL,
                instrumentalness REAL,
                liveness REAL,
                valence REAL,
                tempo REAL,
                time_signature INTEGER,
                track_genre TEXT
            )
        ''')
        conn.commit()

@app.route('/ingest', methods=['POST'])
def ingest_data():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No se recibieron datos"}), 400

    with sqlite3.connect('./Code/Data/spotify_data.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM tracks WHERE track_id = ?", (data['track_id'],))
        exists = cursor.fetchone()

        if exists:
            return jsonify({"status": "info", "message": "Canción ya existe, no se insertó"}), 200

        try:
            cursor.execute('''
                INSERT INTO tracks (
                    track_id, artists, album_name, track_name, popularity,
                    duration_ms, explicit, danceability, energy, key,
                    loudness, mode, speechiness, acousticness, instrumentalness,
                    liveness, valence, tempo, time_signature, track_genre
                ) VALUES (
                    :track_id, :artists, :album_name, :track_name, :popularity,
                    :duration_ms, :explicit, :danceability, :energy, :key,
                    :loudness, :mode, :speechiness, :acousticness, :instrumentalness,
                    :liveness, :valence, :tempo, :time_signature, :track_genre
                )
            ''', data)
            conn.commit()
        except Exception as e:
            logging.error("Error al insertar en la base de datos: %s", e)
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "success", "message": "Canción insertada correctamente"}), 200


if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)
    init_db()
    app.run(debug=True, port=8888)
