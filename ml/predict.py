# ml/predict.py

import joblib
import pandas as pd
from ml.preprocess import preprocess_for_classification


# -----------------------------------
# LOAD CLASSIFICATION MODEL
# -----------------------------------

model = joblib.load("models/logistic_model.pkl")


def predict_student(input_df: pd.DataFrame):
    """
    Predict pass/fail and probability for a single student.

    Args:
        input_df (pd.DataFrame): Raw student input (1 row)

    Returns:
        prediction (int): 0 or 1
        probability (float): Probability of class 1
    """

    # ----------------------------
    # Preprocess input
    # ----------------------------
    processed_input = preprocess_for_classification(input_df)

    # ----------------------------
    # Predict
    # ----------------------------
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    return prediction, probability