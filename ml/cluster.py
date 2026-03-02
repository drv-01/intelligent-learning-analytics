# ml/cluster.py

import joblib
import numpy as np
import pandas as pd


# -----------------------------------
# LOAD CLUSTERING ARTIFACTS
# -----------------------------------

scaler_clustering = joblib.load("models/scaler_clustering.pkl")
pca = joblib.load("models/pca.pkl")
kmeans = joblib.load("models/kmeans.pkl")


cluster_features = [
    "PreviousGrade",
    "AttendanceRate",
    "StudyHoursPerWeek"
]


def cluster_student(input_df: pd.DataFrame):
    """
    Assigns cluster and learner category to student.

    Args:
        input_df (pd.DataFrame): Raw student input (1 row)

    Returns:
        cluster_id (int)
        learner_category (str)
    """

    # -----------------------------------
    # Extract & Clean cluster features
    # -----------------------------------

    # Ensure numeric columns and fill NaNs with 0 (or median, but 0 is safer for now)
    cluster_input = input_df[cluster_features].copy()
    for col in cluster_features:
        cluster_input[col] = pd.to_numeric(cluster_input[col], errors='coerce').fillna(0)

    # -----------------------------------
    # Apply scaling → PCA → KMeans
    # -----------------------------------

    scaled = scaler_clustering.transform(cluster_input)
    pca_transformed = pca.transform(scaled)
    cluster_id = kmeans.predict(pca_transformed)[0]

    # -----------------------------------
    # Dynamic Cluster Ranking
    # -----------------------------------

    # Get centroids in PCA space
    centroids_pca = kmeans.cluster_centers_

    # Convert back to scaled original space
    centroids_scaled = pca.inverse_transform(centroids_pca)

    # Convert back to original feature scale
    centroids_original = scaler_clustering.inverse_transform(centroids_scaled)

    # Compute strength score (sum of 3 academic features)
    cluster_scores = centroids_original.sum(axis=1)

    sorted_clusters = np.argsort(cluster_scores)

    weakest_cluster = sorted_clusters[0]
    strongest_cluster = sorted_clusters[-1]

    if cluster_id == weakest_cluster:
        learner_category = "At Risk"
    elif cluster_id == strongest_cluster:
        learner_category = "High Performer"
    else:
        learner_category = "Average"

    return cluster_id, learner_category