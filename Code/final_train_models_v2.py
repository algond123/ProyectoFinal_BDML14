import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
import logging
import time
from datetime import datetime


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"./DataLogged/{timestamp}_models_train.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']


# Process dataset
def clean_normalize_dataset(input_csv: str, scaled_output_pkl: str, scaler_output_pkl: str):

    start_time = time.time()

    logging.info(f"Clean and Normalize Dataset")

    df = pd.read_csv(input_csv)
    df = df[FEATURES].dropna().copy()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    joblib.dump(scaler, scaler_output_pkl)

    scaled_df = pd.DataFrame(data_scaled, columns=FEATURES)
    joblib.dump(scaled_df, scaled_output_pkl)

    logging.info(f"Scaler saved to: {scaler_output_pkl}")
    logging.info(f"Scaled data saved to: {scaled_output_pkl}")

    logging.info(f"Completed in {time.time() - start_time:.2f} seconds\n\n")

# Kmeans Clustering
def train_kmeans(scaled_input_pkl: str, scaled_labeled_output_pkl: str, model_output_pkl: str):

    start_time = time.time()

    logging.info(f"K-means Training (Get Clusters)")

    data_scaled = joblib.load(scaled_input_pkl)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
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
clean_normalize_dataset('./DataSource/tracks.csv', './DataProduced/data_scaled.pkl', './DataProduced/scaler.pkl')

# Train KMeans using scaled data
train_kmeans('./DataProduced/data_scaled.pkl', './DataProduced/data_scaled_labeled.pkl', './DataProduced/model_kmeans.pkl')

# Save Train and test data using scaled labeled data
save_train_and_test_data('./DataProduced/data_scaled_labeled.pkl', './DataProduced/data_train.pkl', './DataProduced/data_test.pkl')

# Train Random Forest using scaled labeled data
train_random_forest('./DataProduced/data_train.pkl','./DataProduced/model_random_forest.pkl','./DataProduced/grid_random_forest.pkl')

# Train KNN using scaled labeled data
train_knn('./DataProduced/data_train.pkl','./DataProduced/model_knn.pkl','./DataProduced/grid_knn.pkl')

# Train Logistic Regression using scaled labeled data
train_logistic_regression('./DataProduced/data_train.pkl','./DataProduced/model_logistic_regression.pkl','./DataProduced/grid_logistic_regression.pkl')

# Train SVC using scaled labeled data
train_svc('./DataProduced/data_train.pkl','./DataProduced/model_svc.pkl','./DataProduced/grid_svc.pkl')

# Train Gradient Boost using scaled labeled data
train_hist_gradient_boost('./DataProduced/data_train.pkl','./DataProduced/model_hist_gradient_boost.pkl','./DataProduced/grid_hist_gradient_boost.pkl')

# Train XGBoost using scaled labeled data
train_xgboost('./DataProduced/data_train.pkl','./DataProduced/model_xgboost.pkl','./DataProduced/grid_xgboost.pkl')