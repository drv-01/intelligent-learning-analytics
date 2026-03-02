# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml.predict import predict_student
from ml.cluster import cluster_student
from ml.recommend import generate_recommendations


# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI Study Coach - Batch Analytics",
    layout="wide"
)

st.title("🎓 Intelligent Learning Analytics Dashboard")
st.markdown("Upload a student dataset CSV file to analyze academic risk and learner segmentation.")

# -----------------------------------
# REQUIRED COLUMNS
# -----------------------------------

REQUIRED_COLUMNS = [
    "PreviousGrade",
    "AttendanceRate",
    "StudyHoursPerWeek",
    "SleepHours",
    "CommuteTimeMinutes",
    "AssignmentsCompleted",
    "Gender",
    "SubjectStream",
    "ParentalSupport",
    "InternetAccess",
    "FamilyIncomeLevel"
]

# -----------------------------------
# FILE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Student Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # VALIDATE SCHEMA
    # -----------------------------------

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # -----------------------------------
    # RUN ANALYSIS
    # -----------------------------------

    results = []

    for _, row in df.iterrows():

        try:
            student_df = pd.DataFrame([row])

            prediction, probability = predict_student(student_df)
            cluster_id, learner_category = cluster_student(student_df)
            recommendations = generate_recommendations(
                student_df,
                learner_category
            )

            results.append({
                "PredictedPassFail": prediction,
                "Probability": probability,
                "LearnerCategory": learner_category,
                "Recommendations": " | ".join(recommendations)
            })
        except Exception as e:
            st.warning(f"Skipping row due to error: {e}")
            results.append({
                "PredictedPassFail": -1,
                "Probability": 0.0,
                "LearnerCategory": "Unknown",
                "Recommendations": "Error processing this row."
            })

    # Map numeric predictions to descriptive labels
    prediction_labels = []
    for r in results:
        p = r["PredictedPassFail"]
        if p == 1:
            prediction_labels.append("Likely High Performer")
        elif p == 0:
            prediction_labels.append("At Academic Risk")
        else:
            prediction_labels.append("Unknown/Error")

    clean_output = pd.DataFrame({
        "StudentID": df.get("StudentID", df.index),
        "PreviousGrade": df["PreviousGrade"],
        "AttendanceRate": df["AttendanceRate"],
        "StudyHoursPerWeek": df["StudyHoursPerWeek"],
        "SleepHours": df["SleepHours"],
        "CommuteTimeMinutes": df["CommuteTimeMinutes"],
        "AssignmentsCompleted": df["AssignmentsCompleted"],
        "PredictedPassFail": prediction_labels,
        "Probability": [r["Probability"] for r in results],
        "LearnerCategory": [r["LearnerCategory"] for r in results],
        "Recommendations": [r["Recommendations"] for r in results]
    })

    # -----------------------------------
    # DISPLAY RESULTS
    # -----------------------------------

    st.subheader("📊 Prediction Results")
    st.dataframe(clean_output)

    # -----------------------------------
    # SUMMARY INSIGHTS
    # -----------------------------------

    st.divider()
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 Risk Distribution")
        risk_counts = clean_output["PredictedPassFail"].value_counts()
        
        fig1, ax1 = plt.subplots()
        labels = ["At Academic Risk", "Likely High Performer"]
        counts = [
            risk_counts.get("At Academic Risk", 0),
            risk_counts.get("Likely High Performer", 0)
        ]
        ax1.bar(labels, counts, color=['#ff4b4b', '#28a745'])
        ax1.set_ylabel("Number of Students")
        st.pyplot(fig1)

    with col_chart2:
        st.subheader("👥 Learner Category Distribution")
        cluster_counts = clean_output["LearnerCategory"].value_counts()
        
        fig2, ax2 = plt.subplots()
        ax2.bar(cluster_counts.index, cluster_counts.values, color='#0078ff')
        ax2.set_ylabel("Number of Students")
        st.pyplot(fig2)

    # -----------------------------------
    # DOWNLOAD BUTTON
    # -----------------------------------

    st.divider()
    csv = clean_output.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Full Results as CSV",
        data=csv,
        file_name="student_analytics_results.csv",
        mime="text/csv"
    )