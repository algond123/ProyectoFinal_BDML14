# utils/load_csv_to_db.py

import pandas as pd
import sqlite3
import os

# Ruta del CSV y de la base de datos
CSV_PATH = './Code/Source/cancionesSpotify.csv'
DB_PATH = './Code/Data/spotify_data.db'

# Asegurar que el directorio exista
#os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Crear tabla si no existe
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

# Cargar CSV y guardar en la DB
def insert_csv_to_db():
    df = pd.read_csv(CSV_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        for i, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO tracks (
                        track_id, artists, album_name, track_name, popularity,
                        duration_ms, explicit, danceability, energy, key,
                        loudness, mode, speechiness, acousticness, instrumentalness,
                        liveness, valence, tempo, time_signature, track_genre
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['track_id'], row['artists'], row['album_name'], row['track_name'], row['popularity'],
                    row['duration_ms'], row['explicit'], row['danceability'], row['energy'], row['key'],
                    row['loudness'], row['mode'], row['speechiness'], row['acousticness'], row['instrumentalness'],
                    row['liveness'], row['valence'], row['tempo'], row['time_signature'], row['track_genre']
                ))
            except Exception as e:
                print(f"Error en la fila {i}: {e}")
        
        conn.commit()
    print("✅ Ingesta manual completada correctamente.")

# Ejecutar
if __name__ == "__main__":
    init_db()
    insert_csv_to_db()
