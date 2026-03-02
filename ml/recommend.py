# ml/recommend.py

import pandas as pd


def generate_recommendations(input_df: pd.DataFrame, learner_category: str):
    """
    Generate personalized recommendations based on
    learner category and raw student features.

    Args:
        input_df (pd.DataFrame): Raw student input (1 row)
        learner_category (str): "At Risk", "Average", or "High Performer"

    Returns:
        List[str]: Recommendations
    """

    recommendations = []
    row = input_df.iloc[0]

    # ----------------------------
    # Cluster-based guidance
    # ----------------------------
    if learner_category == "High Performer":
        recommendations.append(
            "Encourage advanced coursework and competitive exam preparation."
        )
    elif learner_category == "At Risk":
        recommendations.append(
            "Provide structured academic mentorship and close monitoring."
        )
    else:
        recommendations.append(
            "Improve study consistency and structured planning."
        )

    # ----------------------------
    # Feature-based personalization
    # ----------------------------
    if row["AttendanceRate"] < 75:
        recommendations.append(
            "Increase attendance to improve academic engagement."
        )

    if row["StudyHoursPerWeek"] < 12:
        recommendations.append(
            "Increase weekly study hours with a structured timetable."
        )

    if row["PreviousGrade"] < 65:
        recommendations.append(
            "Revise foundational concepts to strengthen academic base."
        )

    if row["SleepHours"] < 6:
        recommendations.append(
            "Ensure adequate sleep to improve cognitive performance."
        )

    return recommendations