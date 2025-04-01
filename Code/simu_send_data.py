# utils/send_data.py

import pandas as pd
import requests
import time

# Ruta al CSV descargado desde Kaggle
DATASET_PATH = 'data/spotify_dataset.csv'  # Cambia esto si el nombre es diferente

# Cargar el dataset
df = pd.read_csv(DATASET_PATH)

# Enviar los primeros 5 registros al endpoint
for index, row in df.head(5).iterrows():
    json_data = row.to_dict()
    
    print(f"Enviando registro {index + 1}...")
    response = requests.post("http://localhost:5000/ingest", json=json_data)

    try:
        print(response.json())
    except:
        print("Error al interpretar la respuesta del servidor.")
    
    time.sleep(1)  # Esperar 1 segundo entre envíos para simular llegada gradual
