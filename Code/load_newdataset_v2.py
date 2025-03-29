import pandas as pd
import numpy as np

import joblib
import logging
import time
from datetime import datetime


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"./DataLogged/{timestamp}_load_newdataset.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

cluster_to_mood = {
    0: 'Intense',
    1: 'Chill',
    2: 'Melancholic'
}

FEATURES = ['valence','energy','danceability','tempo','loudness']


def predict_moods_from_file(input_data_csv: str, input_model_pkl: str, input_scaler_pkl: str, output_original_predict_pkl: str, output_scaled_predict_pkl: str):

    start_time = time.time()

    logging.info(f"Predict mood for all songs in the CSV dataset")

    # Load input
    df_raw = pd.read_csv(input_data_csv)

    # Drop rows where any FEATURE column is NaN, but keep all other columns
    df = df_raw[df_raw[FEATURES].notna().all(axis=1)].copy()

    # Load scaler and model
    scaler = joblib.load(input_scaler_pkl)
    model = joblib.load(input_model_pkl)

    # Scale only the FEATURE columns
    data_scaled = pd.DataFrame(scaler.transform(df[FEATURES]), columns=FEATURES)

    # Predict moods
    moods = model.predict(data_scaled)

    # Add mood column
    df['mood'] = moods
    df['mood'] = df['mood'].map(cluster_to_mood)
    data_scaled['mood'] = moods

    # Save results
    joblib.dump(df, output_original_predict_pkl)
    joblib.dump(data_scaled, output_scaled_predict_pkl)

    # Logging
    logging.info(f"Mood predictions saved to: {output_original_predict_pkl}")
    logging.info(f"Scaled data with predictions saved to: {output_scaled_predict_pkl}")

    logging.info(f"Completed in {time.time() - start_time:.2f} seconds")


# Predict new database with SVC trained model
predict_moods_from_file('./DataSource/cancionesSpotify.csv', 
                        './DataProduced/model_kmeans.pkl', './DataProduced/scaler.pkl',
                        './DataProduced/newdataset_labeled.pkl', './DataProduced/newdataset_scaled_labeled.pkl')

df = joblib.load('./DataProduced/newdataset_labeled.pkl')
df.to_csv("newdataset_labeled.csv", index=False)

df = joblib.load('./DataProduced/newdataset_scaled_labeled.pkl')
df.to_csv("newdataset_scaled_labeled.csv", index=False)