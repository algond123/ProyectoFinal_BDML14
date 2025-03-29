import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import logging
import time
import joblib


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# CLEAN DATA
def clean_normalize_dataset(input_csv: str, scaled_output_csv: str, scaler_output_pkl: str):

    start_time = time.time()

    logging.info(f"🔄 Clean and Normalize Dataset")

    df = pd.read_csv(input_csv)
    df = df[FEATURES].dropna().copy()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    joblib.dump(scaler, scaler_output_pkl)

    scaled_df = pd.DataFrame(data_scaled, columns=FEATURES)
    scaled_df.to_csv(scaled_output_csv, index=False)

    logging.info(f"Scaler saved to: {scaler_output_pkl}")
    logging.info(f"Scaled data saved to: {scaled_output_csv}")

    logging.info(f"✅ Completed in {time.time() - start_time:.2f} seconds")

# KMEANS CLUSTERING
def train_kmeans(optimal_k: int, scaled_labeled_input_csv: str, scaled_labeled_output_csv: str, kmeans_model_output_pkl: str):

    start_time = time.time()

    logging.info(f"🔄 K-means Training (Get CLusters)")

    data_scaled = pd.read_csv(scaled_labeled_input_csv)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    joblib.dump(kmeans, kmeans_model_output_pkl)

    scaled_df = pd.DataFrame(data_scaled, columns=FEATURES)
    scaled_df.to_csv(scaled_labeled_output_csv, index=False)

    scaled_df['mood'] = labels

    cluster_averages = scaled_df.groupby('mood')[FEATURES].mean()
    logging.info("Average feature values per cluster:\n")
    logging.info(f"\n{cluster_averages}")

    scaled_df.to_csv(scaled_labeled_output_csv, index=False)

    logging.info(f"K-means model saved to: {kmeans_model_output_pkl}")
    logging.info(f"Scaled labeled mood data saved to: {scaled_labeled_output_csv}")
    logging.info(f"✅ Completed in {time.time() - start_time:.2f} seconds")

# RANDOM FOREST MODEL
def train_random_forest(scaled_labeled_input_csv: str, rf_model_output_pkl: str):

    start_time = time.time()

    logging.info(f"🔄 Random Forest Model Training")

    df = pd.read_csv(scaled_labeled_input_csv)
    x = df.drop(columns=["mood"])
    y = df["mood"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
    }

    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, rf_model_output_pkl)

    y_pred = best_model.predict(x_test)

    logging.info(f"Best Parameters (Grid): {grid_search.best_params_}")
    logging.info(f"Best CV Accuracy (Grid): {grid_search.best_score_:.5f}")
    logging.info(f"Test set accuracy (holdout): {accuracy_score(y_test, y_pred):.5f}")
    logging.info(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index = [f"True {i}" for i in range(cm.shape[0])],
        columns = [f"Pred {i}" for i in range(cm.shape[1])]
    )
    logging.info("\nConfusion Matrix:\n" + cm_df.to_string())

    logging.info(f"Random Forest model saved to: {rf_model_output_pkl}")
    logging.info(f"✅ Completed in {time.time() - start_time:.2f} seconds")

# KNN MODEL
def train_knn(scaled_labeled_input_csv: str, knn_model_output_pkl: str):
    start_time = time.time()

    logging.info(f"🔄 KNN Model Training")

    df = pd.read_csv(scaled_labeled_input_csv)
    x = df.drop(columns=["mood"])
    y = df["mood"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [2]
    }

    grid_search = GridSearchCV(
        knn, 
        param_grid, 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, knn_model_output_pkl)

    y_pred = best_model.predict(x_test)

    logging.info(f"Best Parameters (Grid): {grid_search.best_params_}")
    logging.info(f"Best CV Accuracy (Grid): {grid_search.best_score_:.5f}")
    logging.info(f"Test set accuracy (holdout): {accuracy_score(y_test, y_pred):.5f}")
    logging.info(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index = [f"True {i}" for i in range(cm.shape[0])],
        columns = [f"Pred {i}" for i in range(cm.shape[1])]
    )
    logging.info("\nConfusion Matrix:\n" + cm_df.to_string())

    logging.info(f"KNN model saved to: {knn_model_output_pkl}")
    logging.info(f"✅ Completed in {time.time() - start_time:.2f} seconds")

# LOGISTIC REGRESSION MODEL
def train_logistic_regression(scaled_labeled_input_csv: str, lr_model_output_pkl: str):
    start_time = time.time()

    logging.info("🔄 Logistic Regression Model Training")

    df = pd.read_csv(scaled_labeled_input_csv)
    x = df.drop(columns=["mood"])
    y = df["mood"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(max_iter=2000, random_state=42)

    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2'],
    }

    grid_search = GridSearchCV(
        lr,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, lr_model_output_pkl)

    y_pred = best_model.predict(x_test)

    logging.info(f"Best Parameters (Grid): {grid_search.best_params_}")
    logging.info(f"Best CV Accuracy (Grid): {grid_search.best_score_:.5f}")
    logging.info(f"Test set accuracy (holdout): {accuracy_score(y_test, y_pred):.5f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index = [f"True {i}" for i in range(cm.shape[0])],
        columns = [f"Pred {i}" for i in range(cm.shape[1])]
    )
    logging.info("\nConfusion Matrix:\n" + cm_df.to_string())

    logging.info(f"Logistic Regression model saved to: {lr_model_output_pkl}")
    logging.info(f"✅ Completed in {time.time() - start_time:.2f} seconds")

# SVM MODEL
def train_svc(scaled_labeled_input_csv: str, svm_model_output_pkl: str):
    start_time = time.time()

    logging.info("🔄 SVM Model Training")

    df = pd.read_csv(scaled_labeled_input_csv)
    x = df.drop(columns=["mood"])
    y = df["mood"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    svm = SVC(probability=True, random_state=42) 

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale']
    }

    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, svm_model_output_pkl)

    y_pred = best_model.predict(x_test)

    logging.info(f"Best Parameters (Grid): {grid_search.best_params_}")
    logging.info(f"Best CV Accuracy (Grid): {grid_search.best_score_:.5f}")
    logging.info(f"Test set accuracy (holdout): {accuracy_score(y_test, y_pred):.5f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index = [f"True {i}" for i in range(cm.shape[0])],
        columns = [f"Pred {i}" for i in range(cm.shape[1])]
    )
    logging.info("\nConfusion Matrix:\n" + cm_df.to_string())

    logging.info(f"SVC model saved to: {svm_model_output_pkl}")
    logging.info(f"✅ Completed in {time.time() - start_time:.2f} seconds")


# Clean and Normalize dataset
clean_normalize_dataset('tracks.csv', 'spotify_scaled_only.csv', 'scaler.pkl')

# Train KMeans using scaled data
train_kmeans(6, 'spotify_scaled_only.csv', 'spotify_scaled_with_moods.csv', 'spotify_model_kmeans.pkl')

# Train Random Forest using scaled labeled moods
#train_random_forest('spotify_scaled_with_moods.csv','spotify_model_best_random_forest.pkl')

# Train KNN using scaled labeled moods
#train_knn('spotify_scaled_with_moods.csv','spotify_model_best_knn.pkl')

# Train Logistic Regression using scaled labeled moods
train_logistic_regression('spotify_scaled_with_moods.csv','spotify_model_best_logistic_regression.pkl')

# Train SVC using scaled labeled moods
train_svc('spotify_scaled_with_moods.csv','spotify_model_best_svc.pkl')