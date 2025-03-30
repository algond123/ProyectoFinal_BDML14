import pandas as pd
import numpy as np

import joblib
import logging
import time
from datetime import datetime

path_src = './Code/Source/'
path_data = './Code/Data/'
path_models = './Code/Models/'
path_log = './Code/Logging/'

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{path_log}{timestamp}_load_newdataset.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def predict_moods_from_file(input_data_csv: str, input_model_pkl: str, output_original_predict_pkl: str, output_scaled_predict_pkl: str):

    start_time = time.time()

    logging.info(f"Predict mood for all songs in the CSV dataset")

    #Mapping
    cluster_to_mood = {
        0: 'Sadness/Depression',
        1: 'Joy/Excitement',
        2: 'Anger/Tense',
        3: 'Calmness/Relaxation'
    }

    # Load model
    model = joblib.load(input_model_pkl)

    # Features
    old_features = ['energy','tempo','loudness','danceability','valence']
    new_features = ['valence','arousal']

    # Load input
    df_raw = pd.read_csv(input_data_csv)

    # Drop rows where any FEATURE column is NaN, but keep all other columns
    df = df_raw[df_raw[old_features].notna().all(axis=1)].copy()

    # Initialize the data_scaled DataFrame
    data_scaled = df[old_features].dropna().copy()

    # Define scaling ranges for specific features
    scale_ranges = {
        'tempo': (20, 200),
        'loudness': (-60, 0),
        'energy': (0, 1),
        'danceability': (0, 1)
    }

    # Clip the tempo and loudness values to ensure they stay within the expected ranges
    data_scaled['tempo'] = data_scaled['tempo'].clip(lower=20, upper=200)
    data_scaled['loudness'] = data_scaled['loudness'].clip(lower=-60, upper=0)
    data_scaled['energy'] = data_scaled['energy'].clip(lower=0, upper=1)
    data_scaled['danceability'] = data_scaled['danceability'].clip(lower=0, upper=1)

    # Apply MinMax scaling for each feature
    for feature, (min_input, max_input) in scale_ranges.items():
        min_output = 0
        max_output = 1
        data_scaled[feature] = (data_scaled[feature] - min_input) / (max_input - min_input) * (max_output - min_output) + min_output

    # Round the values to 3 decimal places
    data_scaled = data_scaled.round(3)
    data_scaled_save = data_scaled.copy()

    # Log the description after scaling
    #logging.info(f"Scaled Data Description:\n{data_scaled.describe()}")

    # Define the weights for the features
    alpha = 0.5  # Energy has more influence on arousal
    beta = 0.25  # Tempo is still important, but less than energy
    gamma = 0.2  # Loudness is less important, but still relevant
    delta = 0.05  # Danceability plays the least role in arousal
    
    # Calculate the arousal score (weighted sum of selected features)
    data_scaled['arousal'] = (alpha * data_scaled['energy']) + \
                             (beta * data_scaled['tempo']) + \
                             (gamma * data_scaled['loudness']) + \
                             (delta * data_scaled['danceability'])
    
    # Round the values for arousal
    data_scaled = data_scaled.round(3)
    data_scaled_save['arousal'] = data_scaled['arousal']

    # Select the relevant columns (valence and arousal)
    data_scaled = data_scaled[new_features].dropna().copy()

    # Predict moods
    moods = model.predict(data_scaled)

    # Add mood column
    df['mood'] = moods
    df['mood'] = df['mood'].map(cluster_to_mood)

    data_scaled_save['mood'] = moods
    data_scaled_save['track_id'] = df['track_id']

    new_column_order = ['track_id','mood','valence','arousal','energy','tempo','loudness','danceability']
    data_scaled_save = data_scaled_save[new_column_order]

    # Save results
    joblib.dump(df, output_original_predict_pkl)
    joblib.dump(data_scaled_save, output_scaled_predict_pkl)

    # Logging
    logging.info(f"Mood predictions saved to: {output_original_predict_pkl}")
    logging.info(f"Scaled data with predictions saved to: {output_scaled_predict_pkl}")

    logging.info(f"Completed in {time.time() - start_time:.2f} seconds")


# Predict new database with SVC trained model
predict_moods_from_file(f"{path_src}cancionesSpotify.csv",
                        f"{path_models}model_svc.pkl",
                        f"{path_data}newdataset_labeled.pkl",
                        f"{path_data}newdataset_scaled_labeled.pkl")

df = joblib.load(f"{path_data}newdataset_labeled.pkl")
df.to_csv(f"{path_data}newdataset_labeled.csv", index=False)

df = joblib.load(f"{path_data}newdataset_scaled_labeled.pkl")
df.to_csv(f"{path_data}newdataset_scaled_labeled.csv", index=False)