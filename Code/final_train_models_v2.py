import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
import logging
import time
from datetime import datetime

path_src = './Code/Source/'
path_data = './Code/Data/'
path_models = './Code/Models/'
path_log = './Code/Logging/'

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{path_log}{timestamp}_models_train.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)


# Process dataset
def clean_normalize_dataset(input_csv: str, scaled_output_pkl: str):

    start_time = time.time()
    logging.info(f"Clean and Normalize Dataset")

    # Features
    old_features = ['energy','tempo','loudness','danceability','valence']
    new_features = ['valence','arousal']

    # Load the data and clean it (drop missing values)
    df = pd.read_csv(input_csv)
    #logging.info(f"Shape Before: {df.shape}")
    #logging.info(f"Missing data before cleanup: {df[old_features].isnull().sum().sum()} entries.")
    df = df[old_features].dropna().copy()
    #logging.info(f"Shape After: {df.shape}")
    #logging.info(f"Original Data Description:\n{df.describe()}")
    logging.info(f"First few rows of the scaled data:\n{df.head()}")

    # Initialize the data_scaled DataFrame
    data_scaled = df.copy()

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

    # Log the description after calculating arousal
    #logging.info(f"Data with Arousal Score Description:\n{data_scaled.describe()}")
    logging.info(f"First few rows of the scaled data:\n{data_scaled.head()}")

    # Select the relevant columns (valence and arousal)
    scaled_df = data_scaled[new_features].dropna().copy()
    logging.info(f"Shape Save: {df.shape}")

    # Save the cleaned and scaled DataFrame to a pickle file
    scaled_df.to_pickle(scaled_output_pkl)

    logging.info(f"Completed in {time.time() - start_time:.2f} seconds")

# Kmeans Clustering
def train_kmeans(scaled_input_pkl: str, scaled_labeled_output_pkl: str, model_output_pkl: str):

    start_time = time.time()

    logging.info(f"K-means Training (Get Clusters)")

    data_scaled = joblib.load(scaled_input_pkl)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    data_scaled['mood'] = kmeans.fit_predict(data_scaled)

    joblib.dump(kmeans, model_output_pkl)
    joblib.dump(data_scaled, scaled_labeled_output_pkl)

    logging.info(f"Model saved to: {model_output_pkl}")
    logging.info(f"Scaled labeled data saved to: {scaled_labeled_output_pkl}")
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")

# Save Train and Test Vectors
def save_train_and_test_data(scaled_labeled_input_pkl: str, train_data_pkl: str, test_data_pkl: str):

    start_time = time.time()

    logging.info("Splitting and saving train/test data")

    df = joblib.load(scaled_labeled_input_pkl)
    x = df.drop(columns=["mood"])
    y = df["mood"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    joblib.dump((x_train, y_train), train_data_pkl)
    joblib.dump((x_test, y_test), test_data_pkl)

    logging.info(f"Train data saved to: {train_data_pkl}")
    logging.info(f"Test data saved to: {test_data_pkl}")
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")

# RANDOM FOREST MODEL
def train_random_forest(train_data_pkl: str, model_output_pkl: str, grid_output_pkl: str):

    start_time = time.time()

    logging.info(f"Random Forest Model Training")

    x_train, y_train = joblib.load(train_data_pkl)

    model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
    }

    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(x_train, y_train)
    joblib.dump(grid_search, grid_output_pkl)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_output_pkl)

    logging.info(f"Model saved to: {model_output_pkl}")
    logging.info(f"Grid saved to: {grid_output_pkl}")
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")

# KNN MODEL
def train_knn(train_data_pkl: str, model_output_pkl: str, grid_output_pkl: str):
    start_time = time.time()

    logging.info(f"KNN Model Training")

    x_train, y_train = joblib.load(train_data_pkl)

    model = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [2]
    }

    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    joblib.dump(grid_search, grid_output_pkl)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_output_pkl)

    logging.info(f"Model saved to: {model_output_pkl}")
    logging.info(f"Grid saved to: {grid_output_pkl}")
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")

# LOGISTIC REGRESSION MODEL
def train_logistic_regression(train_data_pkl: str, model_output_pkl: str, grid_output_pkl: str):
    start_time = time.time()

    logging.info("Logistic Regression Model Training")

    x_train, y_train = joblib.load(train_data_pkl)

    model = LogisticRegression(max_iter=1000)

    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs'],
        'penalty': ['l2'],
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    joblib.dump(grid_search, grid_output_pkl)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_output_pkl)

    logging.info(f"Model saved to: {model_output_pkl}")
    logging.info(f"Grid saved to: {grid_output_pkl}")
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")

# SVC MODEL
def train_svc(train_data_pkl: str, model_output_pkl: str, grid_output_pkl: str):
    start_time = time.time()

    logging.info("SVC Model Training")

    x_train, y_train = joblib.load(train_data_pkl)

    model = SVC(probability=False) 

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear']
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    joblib.dump(grid_search, grid_output_pkl)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_output_pkl)

    logging.info(f"Model saved to: {model_output_pkl}")
    logging.info(f"Grid saved to: {grid_output_pkl}")
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")

# HIST GRADIENT BOOST MODEL
def train_hist_gradient_boost(train_data_pkl: str, model_output_pkl: str, grid_output_pkl: str):
    start_time = time.time()
    logging.info("Hist Gradient Boosting Model Training")

    x_train, y_train = joblib.load(train_data_pkl)

    model = HistGradientBoostingClassifier(random_state=42)

    param_grid = {
        'max_iter': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    joblib.dump(grid_search, grid_output_pkl)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_output_pkl)

    logging.info(f"Model saved to: {model_output_pkl}")
    logging.info(f"Grid saved to: {grid_output_pkl}")
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")

# XGBOOST MODEL
def train_xgboost(train_data_pkl: str, model_output_pkl: str, grid_output_pkl: str):
    start_time = time.time()

    logging.info("XGBoost Model Training")

    x_train, y_train = joblib.load(train_data_pkl)

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    joblib.dump(grid_search, grid_output_pkl)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_output_pkl)

    logging.info(f"Model saved to: {model_output_pkl}")
    logging.info(f"Grid saved to: {grid_output_pkl}")
    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")


# Clean and Normalize dataset
clean_normalize_dataset(f"{path_src}tracks.csv",
                        f"{path_data}data_scaled.pkl")

#'''
# Train KMeans using scaled data
train_kmeans(f"{path_data}data_scaled.pkl",
             f"{path_data}data_scaled_labeled.pkl",
             f"{path_models}model_kmeans.pkl")

# Save Train and test data using scaled labeled data
save_train_and_test_data(f"{path_data}data_scaled_labeled.pkl", 
                         f"{path_data}data_train.pkl",
                         f"{path_data}data_test.pkl")

# Train Random Forest using scaled labeled data
train_random_forest(f"{path_data}data_train.pkl",
                    f"{path_models}model_random_forest.pkl",
                    f"{path_models}grid_random_forest.pkl")

# Train KNN using scaled labeled data
train_knn(f"{path_data}data_train.pkl",
          f"{path_models}model_knn.pkl",
          f"{path_models}grid_knn.pkl")

# Train Logistic Regression using scaled labeled data
train_logistic_regression(f"{path_data}data_train.pkl",
                          f"{path_models}model_logistic_regression.pkl",
                          f"{path_models}grid_logistic_regression.pkl")

# Train SVC using scaled labeled data
train_svc(f"{path_data}data_train.pkl",
          f"{path_models}model_svc.pkl",
          f"{path_models}grid_svc.pkl")

# Train Gradient Boost using scaled labeled data
train_hist_gradient_boost(f"{path_data}data_train.pkl",
                          f"{path_models}model_hist_gradient_boost.pkl",
                          f"{path_models}grid_hist_gradient_boost.pkl")

# Train XGBoost using scaled labeled data
train_xgboost(f"{path_data}data_train.pkl",
              f"{path_models}model_xgboost.pkl",
              f"{path_models}grid_xgboost.pkl")
#'''