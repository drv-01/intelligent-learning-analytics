# ml/preprocess.py

import pandas as pd
import joblib


# -----------------------------------
# LOAD TRAINED ARTIFACTS
# -----------------------------------

# Classification scaler
scaler_classification = joblib.load("models/scaler_classification.pkl")

# Load model only to retrieve training feature names
model = joblib.load("models/logistic_model.pkl")


# These must match exactly what was used during training
continuous_features_classification = [
    "PreviousGrade",
    "AttendanceRate",
    "StudyHoursPerWeek",
    "SleepHours",
    "CommuteTimeMinutes",
    "AssignmentsCompleted"
]


# -----------------------------------
# CLASSIFICATION PREPROCESSING
# -----------------------------------

def preprocess_for_classification(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares raw student input for classification model.

    Steps:
    1. Ensure all expected training features exist (fill missing with 0)
    2. One-hot encode categorical variables
    3. Re-align with model feature names
    4. Scale continuous numeric features
    """

    # Create a copy to avoid modifying original
    df = input_df.copy()

    # 1. One-hot encode categoricals
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 2. Align with training features
    model_features = model.feature_names_in_

    # Ensure every feature used during training is present
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure correct column order and discard extra columns
    df_final = df_encoded[model_features].copy()

    # 3. Scale numeric features
    # Ensure they are numeric to avoid errors
    for col in continuous_features_classification:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

    df_final[continuous_features_classification] = (
        scaler_classification.transform(
            df_final[continuous_features_classification]
        )
    )

    return df_final