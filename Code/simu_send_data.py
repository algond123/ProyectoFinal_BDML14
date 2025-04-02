# utils/send_data.py

import pandas as pd
import requests
import time

df = pd.read_csv('./Code/Source/cancionesSpotify.csv' )

for index, row in df.head(5).iterrows():
    json_data = row.to_dict()
    
    print(f"Enviando registro {index + 1}...")
    response = requests.post("http://localhost:5000/ingest", json=json_data)

    try:
        print(response.json())
    except:
        print("Error al interpretar la respuesta del servidor.")
    
    time.sleep(1)
