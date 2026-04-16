import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml.predict import predict_student
from ml.cluster import cluster_student
from ml.recommend import generate_recommendations
from agent.coach_agent import get_coach_agent
from langchain_core.messages import HumanMessage

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI Study Coach",
    layout="wide"
)

st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Workflow", ["Batch Analytics", "Interactive AI Coach"])

# -----------------------------------
# REQUIRED COLUMNS
# -----------------------------------
REQUIRED_COLUMNS = [
    "PreviousGrade", "AttendanceRate", "StudyHoursPerWeek", "SleepHours",
    "CommuteTimeMinutes", "AssignmentsCompleted", "Gender", "SubjectStream",
    "ParentalSupport", "InternetAccess", "FamilyIncomeLevel"
]

if app_mode == "Batch Analytics":
    st.title("🎓 Intelligent Learning Analytics Dashboard")
    st.markdown("Upload a student dataset CSV file to analyze academic risk and learner segmentation.")

    uploaded_file = st.file_uploader("Upload Student Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Dataset Preview")
        st.dataframe(df.head())

        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()

        results = []
        for _, row in df.iterrows():
            try:
                student_df = pd.DataFrame([row])
                prediction, probability = predict_student(student_df)
                cluster_id, learner_category = cluster_student(student_df)
                recommendations = generate_recommendations(student_df, learner_category)

                results.append({
                    "PredictedPassFail": prediction,
                    "Probability": probability,
                    "LearnerCategory": learner_category,
                    "Recommendations": " | ".join(recommendations)
                })
            except Exception as e:
                results.append({
                    "PredictedPassFail": -1, "Probability": 0.0,
                    "LearnerCategory": "Unknown", "Recommendations": "Error processing this row."
                })

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

        st.subheader("📊 Prediction Results")
        st.dataframe(clean_output)

        st.divider()
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("📈 Risk Distribution")
            risk_counts = clean_output["PredictedPassFail"].value_counts()
            fig1, ax1 = plt.subplots()
            labels = ["At Academic Risk", "Likely High Performer"]
            counts = [risk_counts.get("At Academic Risk", 0), risk_counts.get("Likely High Performer", 0)]
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

        st.divider()
        csv = clean_output.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Results as CSV",
            data=csv,
            file_name="student_analytics_results.csv",
            mime="text/csv"
        )

elif app_mode == "Interactive AI Coach":
    st.title("🤖 AI Study Coach")
    st.markdown("I reason about student performance, plan learning paths, and retrieve external tutorials.")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm your AI Study Coach. Please describe the student's performance or goals, and I will analyze the gaps, plan a strategy, and find tutorials."}
        ]
        
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Enter student context, problem, or question..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Reasoning through the student's condition, calling tools (Web/RAG), and drafting a plan..."):
                try:
                    agent = get_coach_agent()
                    config = {"configurable": {"thread_id": "session_default"}}
                    
                    response = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config)
                    
                    # Output the final response generated by the Agent
                    model_out = response["messages"][-1].content
                    st.markdown(model_out)
                    st.session_state["messages"].append({"role": "assistant", "content": model_out})
                    
                except Exception as e:
                    st.error(f"Agent Execution Error: {e}")