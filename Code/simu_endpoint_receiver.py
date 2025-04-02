
from flask import Flask, request, jsonify
import sqlite3
import os
import json

app = Flask(__name__)

# Crear tabla si no existe
def init_db():
    with sqlite3.connect('./Code/Data/spotify_data.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                name TEXT,
                popularity INTEGER,
                duration_ms INTEGER,
                explicit INTEGER,
                artists TEXT,
                release_date TEXT,
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
                time_signature INTEGER
            )
        ''')
        conn.commit()

@app.route('/ingest', methods=['POST'])
def ingest_data():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No se recibieron datos"}), 400

    # Insertar en SQLite
    with sqlite3.connect('./Code/Data/spotify_data.db') as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO tracks VALUES (
                    :id, :name, :popularity, :duration_ms, :explicit, :artists, :release_date,
                    :danceability, :energy, :key, :loudness, :mode, :speechiness, :acousticness,
                    :instrumentalness, :liveness, :valence, :tempo, :time_signature
                )
            ''', data)
            conn.commit()
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "success", "message": "Datos guardados en base de datos"}), 200

if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)
    init_db()
    app.run(debug=True, port=5000)
