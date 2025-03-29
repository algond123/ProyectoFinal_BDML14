import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import joblib
import logging
import time
from datetime import datetime

path_src = './Code/Source/'
path_data = './Code/Data/'
path_models = './Code/Models/'
path_log = './Code/Logging/'

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{path_log}{timestamp}_models_report.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

FEATURES = ['valence','energy','danceability','tempo','loudness']


def clusters_info(scaled_pkl: str, model_pkl: str):

    start_time = time.time()

    logging.info(f"Clusters Information")

    # Load model and data
    model = joblib.load(model_pkl)
    df = joblib.load(scaled_pkl)

    # Predict clusters if not already labeled
    df['mood'] = model.predict(df)

    # Cluster sizes
    cluster_counts = df['mood'].value_counts().sort_index()
    # Feature averages per cluster
    cluster_summary = df.groupby('mood')[FEATURES].mean()

    # Logging
    logging.info("Cluster Sizes:\n" + cluster_counts.to_string())
    logging.info("Cluster Feature Averages:\n" + cluster_summary.round(5).to_string())
    logging.info(f"Clusters Information completed in {time.time() - start_time:.2f} seconds \n\n")

def evaluate_model(test_data_pkl: str, model_pkl: str, grid_pkl: str, model_name: str):

    start_time = time.time()

    logging.info(f"Evaluating {model_name} on test data")

    # Load model, grid and test data
    model = joblib.load(model_pkl)
    grid = joblib.load(grid_pkl)

    # Predict
    x_test, y_test = joblib.load(test_data_pkl)
    y_pred = model.predict(x_test)

    best_params = grid.best_params_
    best_cv_acc = grid.best_score_
    test_acc = accuracy_score(y_test, y_pred)

    # Log model parameters and scores
    logging.info(f"{model_name} | Best Params: {best_params}")
    logging.info(f"{model_name} | Best Grid Accuracy: {best_cv_acc:.5f}")
    logging.info(f"{model_name} | Test Accuracy: {test_acc:.5f}")

    # Classification Report
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {i}" for i in range(cm.shape[0])],
        columns=[f"Pred {i}" for i in range(cm.shape[1])]
    )
    logging.info("Confusion Matrix:\n" + cm_df.to_string())

    logging.info(f"{model_name} evaluation completed in {time.time() - start_time:.2f} seconds \n\n")

    return {
        "Model": model_name,
        "Best Grid Accuracy": round(best_cv_acc, 5),
        "Test Accuracy": round(test_acc, 5)
    }


results = []

#Kmeans
#clusters_info("{path_models}data_scaled.pkl","{path_models}model_kmeans.pkl")

#'''
#Random Forest
results.append(evaluate_model(f"{path_data}data_test.pkl",
                              f"{path_models}model_random_forest.pkl",
                              f"{path_models}grid_random_forest.pkl",
                              f"Random Forest"))

#KNN
results.append(evaluate_model(f"{path_data}data_test.pkl",
                              f"{path_models}model_knn.pkl",
                              f"{path_models}grid_knn.pkl",
                              f"KNN"))

#Logistic Regression
results.append(evaluate_model(f"{path_data}data_test.pkl",
                              f"{path_models}model_logistic_regression.pkl",
                              f"{path_models}grid_logistic_regression.pkl",
                              f"Logistic Regression"))

#SVC
results.append(evaluate_model(f"{path_data}data_test.pkl",
                              f"{path_models}model_svc.pkl",
                              f"{path_models}grid_svc.pkl",
                              f"SVC"))

#Hist Gradient Boost
results.append(evaluate_model(f"{path_data}data_test.pkl",
                              f"{path_models}model_hist_gradient_boost.pkl",
                              f"{path_models}grid_hist_gradient_boost.pkl",
                              f"Hist Gradient Boost"))

#XGBoost
results.append(evaluate_model(f"{path_data}data_test.pkl",
                              f"{path_models}model_xgboost.pkl",
                              f"{path_models}grid_xgboost.pkl",
                              f"XGBoost"))

summary_df = pd.DataFrame(results)
logging.info(f"\nModel Evaluation Summary:\n{summary_df.to_string(index=False)}\n")
#'''