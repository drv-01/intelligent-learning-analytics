import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time

from ml.predict import predict_student
from ml.cluster import cluster_student
from ml.recommend import generate_recommendations
from agent.coach_agent import get_coach_agent
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────
# PAGE CONFIG & DARK THEME STYLING
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Coach | Learning Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── ROOT / GLOBAL ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0f18 !important;
    color: #e2e8f0 !important;
}

/* Main background */
.stApp { background-color: #0d0f18; }

/* ─── SIDEBAR ───────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d0f18 100%) !important;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
}

/* ─── HEADER BANNER ─────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2847 40%, #1a1040 100%);
    border: 1px solid #2563eb33;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #3b82f640 0%, transparent 70%);
    border-radius: 50%;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 150px; height: 150px;
    background: radial-gradient(circle, #8b5cf640 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.main-header p { color: #94a3b8; font-size: 1rem; margin: 0; }

/* ─── METRIC CARDS ──────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #3b82f6;
}
.metric-card .metric-value {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .metric-label {
    color: #64748b; font-size: 0.82rem; margin-top: 0.3rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* ─── SECTION HEADERS ───────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.6rem 0; margin: 1.2rem 0 0.8rem 0;
    border-bottom: 1px solid #1e293b;
}
.section-header h3 {
    font-size: 1.1rem; font-weight: 700;
    color: #e2e8f0; margin: 0;
}

/* ─── CHART CONTAINERS ──────────────────────────── */
.chart-container {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.2rem;
    transition: border-color 0.2s;
}
.chart-container:hover { border-color: #3b82f666; }

/* ─── PREDICTION TABLE ──────────────────────────── */
.dataframe th {
    background: #1e3a5f !important;
    color: #60a5fa !important;
    font-weight: 600;
}
.dataframe td { color: #cbd5e1 !important; }

/* ─── BADGE PILLS ───────────────────────────────── */
.badge-risk { background:#7f1d1d; color:#fca5a5; padding:3px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.badge-pass { background:#14532d; color:#86efac; padding:3px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.badge-avg  { background:#1e3a5f; color:#93c5fd; padding:3px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }

/* ─── CHAT INTERFACE ────────────────────────────── */
.chat-container {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    min-height: 120px;
}
.stChatMessage { background: #1e293b !important; border-radius: 12px !important; }

/* ─── REASONING STEPS ───────────────────────────── */
.reasoning-step {
    background: #0f2044;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #93c5fd;
}

/* ─── STRUCTURED OUTPUT CARDS ───────────────────── */
.diagnosis-card {
    background: linear-gradient(135deg, #1f1535, #0d1a2e);
    border: 1px solid #6d28d966;
    border-radius: 14px;
    padding: 1.4rem;
    margin: 0.8rem 0;
}
.plan-card {
    background: linear-gradient(135deg, #0d2a1a, #0a1f2e);
    border: 1px solid #16a34a66;
    border-radius: 14px;
    padding: 1.4rem;
    margin: 0.8rem 0;
}
.resources-card {
    background: linear-gradient(135deg, #1a1f0d, #1e2a0d);
    border: 1px solid #ca8a0466;
    border-radius: 14px;
    padding: 1.4rem;
    margin: 0.8rem 0;
}
.insight-card {
    background: linear-gradient(135deg, #1a0d20, #0d182e);
    border: 1px solid #ec489966;
    border-radius: 14px;
    padding: 1.4rem;
    margin: 0.8rem 0;
}

/* ─── SESSION MEMORY PANEL ──────────────────────── */
.memory-item {
    background: #1e293b;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    border-left: 3px solid #8b5cf6;
    font-size: 0.85rem;
    color: #cbd5e1;
}

/* ─── DOWNLOAD BUTTON ───────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover { transform: translateY(-2px) !important; opacity: 0.9 !important; }

/* ─── ALERTS ──────────────────────────────────────  */
.stAlert { border-radius: 10px !important; }

/* ─── FILE UPLOADER ─────────────────────────────── */
.stFileUploader { background: #111827 !important; border-radius: 12px !important; border: 1px dashed #334155 !important; }

/* ─── RADIO BUTTONS ─────────────────────────────── */
.stRadio > label { color: #94a3b8 !important; font-size: 0.85rem; }

/* ─── INPUTS ────────────────────────────────────── */
.stChatInput > div { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 12px !important; }
.stChatInput input { color: #e2e8f0 !important; }

/* ─── PROGRESS BAR ──────────────────────────────── */
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; }

/* ─── DIVIDER ───────────────────────────────────── */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# REQUIRED COLUMNS
# ─────────────────────────────────────────
REQUIRED_COLUMNS = [
    "PreviousGrade", "AttendanceRate", "StudyHoursPerWeek", "SleepHours",
    "CommuteTimeMinutes", "AssignmentsCompleted", "Gender", "SubjectStream",
    "ParentalSupport", "InternetAccess", "FamilyIncomeLevel"
]

# ─────────────────────────────────────────
# PERSISTENT DATA HELPERS (NEW)
# ─────────────────────────────────────────
UPLOAD_PATH = "data/uploaded_student_data.csv"

def save_uploaded_data(df):
    df.to_csv(UPLOAD_PATH, index=False)

def load_persisted_data():
    if os.path.exists(UPLOAD_PATH):
        try:
            return pd.read_csv(UPLOAD_PATH)
        except Exception:
            return None
    return None


# ─────────────────────────────────────────
# MATPLOTLIB DARK THEME HELPER
# ─────────────────────────────────────────
def apply_dark_style(fig, ax):
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem;'>🎓</div>
        <div style='font-size:1.1rem; font-weight:700; color:#60a5fa;'>AI Study Coach</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:0.2rem;'>Intelligent Learning Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    app_mode = st.radio(
        "Navigation",
        ["📊 Batch Analytics", "🤖 AI Coach", "🧠 Session Memory"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; padding: 0.5rem 0; text-align:center;'>
        Built with ❤️ for Educational Excellence
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 1: BATCH ANALYTICS DASHBOARD
# ─────────────────────────────────────────
if app_mode == "📊 Batch Analytics":
    
    st.markdown("""
    <div class='main-header'>
        <h1>🎓 Intelligent Learning Analytics Dashboard</h1>
        <p>Upload a student dataset to analyze academic risk, learner segmentation, and personalized recommendations at scale.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state from disk if not already set
    if "uploaded_df" not in st.session_state:
        persisted = load_persisted_data()
        if persisted is not None:
            st.session_state["uploaded_df"] = persisted


    uploaded_file = st.file_uploader(
        "Upload Student Dataset (CSV)",
        type=["csv"],
        help="Ensure columns: PreviousGrade, AttendanceRate, StudyHoursPerWeek, SleepHours, CommuteTimeMinutes, AssignmentsCompleted, Gender, SubjectStream, ParentalSupport, InternetAccess, FamilyIncomeLevel"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_df"] = df
        save_uploaded_data(df)
        st.success(f"✅ Dataset saved successfully to {UPLOAD_PATH}")

        
        # ── Dataset Preview
        st.markdown("<div class='section-header'><h3>📄 Dataset Preview</h3></div>", unsafe_allow_html=True)
        st.dataframe(df.head(5), use_container_width=True)

        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        # ── Process all students
        results = []
        progress_bar = st.progress(0, text="Analyzing students...")
        total = len(df)
        
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                student_df = pd.DataFrame([row])
                prediction, probability = predict_student(student_df)
                cluster_id, learner_category = cluster_student(student_df)
                recommendations = generate_recommendations(student_df, learner_category)
                results.append({
                    "PredictedPassFail": prediction,
                    "Probability": round(probability * 100, 1),
                    "LearnerCategory": learner_category,
                    "Recommendations": " | ".join(recommendations)
                })
            except Exception:
                results.append({
                    "PredictedPassFail": -1, "Probability": 0.0,
                    "LearnerCategory": "Unknown", "Recommendations": "Error processing this row."
                })
            progress_bar.progress((i + 1) / total, text=f"Analyzing student {i+1}/{total}...")
        
        progress_bar.empty()

        # Label mapping
        prediction_labels = []
        for r in results:
            p = r["PredictedPassFail"]
            if p == 1:   prediction_labels.append("Likely High Performer")
            elif p == 0: prediction_labels.append("At Academic Risk")
            else:        prediction_labels.append("Unknown/Error")

        clean_output = pd.DataFrame({
            "StudentID":           df.get("StudentID", pd.Series(range(1, len(df)+1))),
            "PreviousGrade":       df["PreviousGrade"],
            "AttendanceRate":      df["AttendanceRate"],
            "StudyHoursPerWeek":   df["StudyHoursPerWeek"],
            "SleepHours":          df["SleepHours"],
            "PredictedPassFail":   prediction_labels,
            "Probability (%)":     [r["Probability"] for r in results],
            "LearnerCategory":     [r["LearnerCategory"] for r in results],
            "Recommendations":     [r["Recommendations"] for r in results]
        })

        # ── KPI METRIC CARDS
        total_students   = len(clean_output)
        at_risk_count    = (clean_output["PredictedPassFail"] == "At Academic Risk").sum()
        high_perf_count  = (clean_output["PredictedPassFail"] == "Likely High Performer").sum()
        avg_prob         = clean_output["Probability (%)"].mean()
        avg_attendance   = df["AttendanceRate"].mean()
        avg_study_hrs    = df["StudyHoursPerWeek"].mean()

        st.markdown("<div class='section-header'><h3>📊 Key Performance Indicators</h3></div>", unsafe_allow_html=True)
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        
        def kpi_card(col, value, label):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)
        
        kpi_card(kpi1, total_students, "Total Students")
        kpi_card(kpi2, at_risk_count, "At Academic Risk")
        kpi_card(kpi3, high_perf_count, "High Performers")
        kpi_card(kpi4, f"{avg_prob:.1f}%", "Avg Success Prob")
        kpi_card(kpi5, f"{avg_attendance:.1f}%", "Avg Attendance")
        kpi_card(kpi6, f"{avg_study_hrs:.1f}h", "Avg Study Hours/Wk")

        # ── FULL RESULTS TABLE
        st.markdown("<div class='section-header'><h3>🗂️ Full Prediction Results</h3></div>", unsafe_allow_html=True)
        st.dataframe(clean_output, use_container_width=True, height=300)

        # ── CHARTS ROW 1
        st.markdown("<div class='section-header'><h3>📈 Analytics Visualizations</h3></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        # Chart 1: Risk Distribution Bar
        with col1:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(4, 3.2))
            apply_dark_style(fig1, ax1)
            risk_counts = clean_output["PredictedPassFail"].value_counts()
            labels = ["At Academic Risk", "Likely High Performer"]
            counts = [risk_counts.get("At Academic Risk", 0), risk_counts.get("Likely High Performer", 0)]
            colors = ["#ef4444", "#22c55e"]
            bars = ax1.bar(["At Risk", "High Performer"], counts, color=colors, width=0.5, edgecolor='none', linewidth=0)
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         str(count), ha='center', va='bottom', color='#e2e8f0', fontsize=11, fontweight='700')
            ax1.set_ylabel("Students", color='#94a3b8')
            ax1.set_title("Risk Distribution", color='#e2e8f0', fontweight='700', pad=10)
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        # Chart 2: Learner Category Donut
        with col2:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(4, 3.2))
            apply_dark_style(fig2, ax2)
            cat_counts = clean_output["LearnerCategory"].value_counts()
            pie_colors = ["#3b82f6", "#f59e0b", "#10b981"]
            wedge_props = dict(width=0.5, edgecolor='#111827', linewidth=2)
            ax2.pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%',
                    colors=pie_colors[:len(cat_counts)], wedgeprops=wedge_props,
                    textprops={'color': '#e2e8f0', 'fontsize': 9})
            ax2.set_title("Learner Categories", color='#e2e8f0', fontweight='700', pad=10)
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        # Chart 3: Probability Distribution Histogram
        with col3:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(4, 3.2))
            apply_dark_style(fig3, ax3)
            probs = clean_output["Probability (%)"]
            ax3.hist(probs, bins=15, color='#6366f1', alpha=0.85, edgecolor='#111827', linewidth=0.5)
            ax3.axvline(probs.mean(), color='#f59e0b', linestyle='--', linewidth=1.5, label=f'Mean: {probs.mean():.1f}%')
            ax3.set_xlabel("Success Probability (%)", color='#94a3b8')
            ax3.set_ylabel("Students", color='#94a3b8')
            ax3.set_title("Probability Distribution", color='#e2e8f0', fontweight='700', pad=10)
            ax3.legend(framealpha=0, labelcolor='#f59e0b', fontsize=8)
            st.pyplot(fig3)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── CHARTS ROW 2
        col4, col5 = st.columns(2)

        # Chart 4: Scatter – Study Hours vs Grade
        with col4:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            fig4, ax4 = plt.subplots(figsize=(5, 3.5))
            apply_dark_style(fig4, ax4)
            color_map = {"At Academic Risk": "#ef4444", "Likely High Performer": "#22c55e", "Unknown/Error": "#64748b"}
            for label_v, grp in clean_output.groupby("PredictedPassFail"):
                idx = grp.index
                ax4.scatter(df.loc[idx, "StudyHoursPerWeek"], df.loc[idx, "PreviousGrade"],
                            c=color_map.get(label_v, "#64748b"), alpha=0.7, s=30,
                            label=label_v, edgecolors='none')
            ax4.set_xlabel("Study Hours/Week", color='#94a3b8')
            ax4.set_ylabel("Previous Grade", color='#94a3b8')
            ax4.set_title("Study Hours vs. Previous Grade", color='#e2e8f0', fontweight='700', pad=10)
            ax4.legend(framealpha=0, labelcolor='#94a3b8', fontsize=7.5)
            st.pyplot(fig4)
            st.markdown("</div>", unsafe_allow_html=True)

        # Chart 5: Attendance Rate by Learner Category
        with col5:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            fig5, ax5 = plt.subplots(figsize=(5, 3.5))
            apply_dark_style(fig5, ax5)
            merged = clean_output.copy()
            merged["AttendanceRate"] = df["AttendanceRate"].values
            grouped_att = merged.groupby("LearnerCategory")["AttendanceRate"].mean()
            bar_colors = {"At Risk": "#ef4444", "Average": "#f59e0b", "High Performer": "#22c55e"}
            bars5 = ax5.barh(grouped_att.index, grouped_att.values,
                             color=[bar_colors.get(k, "#6366f1") for k in grouped_att.index],
                             edgecolor='none', height=0.5)
            for bar, val in zip(bars5, grouped_att.values):
                ax5.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                         f"{val:.1f}%", va='center', color='#e2e8f0', fontsize=9, fontweight='600')
            ax5.set_xlabel("Avg Attendance Rate (%)", color='#94a3b8')
            ax5.set_title("Avg Attendance by Learner Category", color='#e2e8f0', fontweight='700', pad=10)
            ax5.set_xlim(0, 110)
            st.pyplot(fig5)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── DOWNLOAD BUTTON
        st.markdown("<div class='section-header'><h3>⬇️ Export Results</h3></div>", unsafe_allow_html=True)
        csv_data = clean_output.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Analytics Report (CSV)",
            data=csv_data,
            file_name="student_analytics_results.csv",
            mime="text/csv",
            use_container_width=False
        )
    else:
        # Landing state with instructions
        st.markdown("""
        <div style='background: linear-gradient(135deg,#0f172a,#111827); border:1px dashed #334155; 
                    border-radius:16px; padding:3rem; text-align:center; margin-top:1rem;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>📂</div>
            <div style='font-size:1.3rem; font-weight:700; color:#e2e8f0; margin-bottom:0.5rem;'>Upload a Student Dataset to Begin</div>
            <div style='color:#64748b; font-size:0.9rem; max-width:500px; margin:0 auto;'>
                Upload a CSV file with student performance data. Required columns include PreviousGrade, 
                AttendanceRate, StudyHoursPerWeek, SleepHours, CommuteTimeMinutes, AssignmentsCompleted,
                Gender, SubjectStream, ParentalSupport, InternetAccess, and FamilyIncomeLevel.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# TAB 2: INTERACTIVE AI COACH
# ─────────────────────────────────────────
elif app_mode == "🤖 AI Coach":
    
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 Autonomous AI Study Coach</h1>
        <p>Powered by LangGraph · Chain-of-Thought Reasoning · ChromaDB RAG · Session Memory · Web Search</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STUDENT DATA SELECTOR (NEW)
    selected_student_context = ""
    if "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]
        st.markdown("<div class='section-header'><h3>📋 Select Student for Diagnosis</h3></div>", unsafe_allow_html=True)
        
        # Use StudentID if exists, otherwise use Index
        id_col = "StudentID" if "StudentID" in df.columns else None
        student_options = [(i, f"Student {df.loc[i, id_col]}" if id_col else f"Student {i}") for i in df.index]
        
        selected_idx = st.selectbox(
            "Which student should the Coach analyze?",
            options=[opt[0] for opt in student_options],
            format_func=lambda x: dict(student_options)[x],
            help="Select a student from the uploaded dataset for personalized coaching."
        )
        
        if selected_idx is not None:
            student_data = df.loc[selected_idx].to_dict()
            selected_student_context = f"\n\nCURRENT STUDENT DATA:\n{str(student_data)}"
            
            with st.expander("🔍 View Selected Student Data"):
                st.json(student_data)
    else:
        st.info("💡 Tip: Upload a dataset in the **Batch Analytics** tab to analyze specific students.")


    # ── SESSION STATE INIT
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": (
                "👋 Hello! I'm your **Autonomous AI Study Coach**.\n\n"
                "I use **LangGraph multi-step reasoning**, **ChromaDB RAG** for tutorial retrieval, "
                "and **live web search** to diagnose your learning gaps and build personalized plans.\n\n"
                "**Try asking me things like:**\n"
                "- *'I'm struggling with math and my attendance is low. Help me plan.'*\n"
                "- *'A student scored 55% with 10 study hours/week — what should they do?'*\n"
                "- *'Create a 4-week study plan for an at-risk high schooler.'*\n\n"
                "I'll reason through the problem, retrieve resources, and output a structured Diagnosis + Plan."
            )}
        ]
    if "session_steps" not in st.session_state:
        st.session_state["session_steps"] = []

    # ── DISPLAY CHAT HISTORY
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── CHAT INPUT
    if prompt := st.chat_input("Describe the student's situation, challenges, or goals..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Show reasoning pipeline progress
            pipeline_placeholder = st.empty()
            
            pipeline_steps = [
                ("🧠", "Step 1: Chain-of-Thought Reasoning — Analyzing learning gaps..."),
                ("📦", "Step 2: RAG Retrieval — Fetching tutorials from ChromaDB..."),
                ("🌐", "Step 3: Web Search — Finding external resources..."),
                ("📋", "Step 4: Generating structured Diagnosis + Plan..."),
            ]
            
            for icon, step_text in pipeline_steps:
                pipeline_placeholder.markdown(f"""
                <div class='reasoning-step'>{icon} {step_text}</div>
                """, unsafe_allow_html=True)
                time.sleep(0.4)
            
            pipeline_placeholder.empty()
            
            with st.spinner("🤖 Finalizing personalized coaching report..."):
                try:
                    agent = get_coach_agent()
                    config = {"configurable": {"thread_id": "session_default"}}
                    
                    response = agent.invoke(
                        {
                            "messages": [HumanMessage(content=prompt + selected_student_context)],
                            "reasoning_steps": [],
                            "tool_results": [],
                            "final_response": ""
                        },
                        config
                    )
                    
                    final_output = response.get("final_response", "")
                    reasoning   = response.get("reasoning_steps", [])
                    
                    if not final_output and response.get("messages"):
                        for msg_obj in reversed(response["messages"]):
                            if hasattr(msg_obj, "content") and msg_obj.content:
                                final_output = msg_obj.content
                                break
                    
                    # ── Show Reasoning Steps expander
                    if reasoning:
                        with st.expander("🧠 View Chain-of-Thought Reasoning Steps", expanded=False):
                            for step in reasoning:
                                if step.strip():
                                    st.markdown(f"""
                                    <div class='reasoning-step'>{step}</div>
                                    """, unsafe_allow_html=True)
                    
                    # ── Parse and display structured output sections
                    sections = {
                        "diagnosis": "",
                        "plan": "",
                        "resources": "",
                        "insight": ""
                    }
                    
                    current_section = None
                    if isinstance(final_output, list):
                        final_output = "\n".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in final_output])
                    lines = final_output.split("\n")
                    buffer = []
                    
                    for line in lines:
                        low = line.lower()
                        if "## 🔍 diagnosis" in low or "## diagnosis" in low:
                            if current_section and buffer:
                                sections[current_section] = "\n".join(buffer).strip()
                                buffer = []
                            current_section = "diagnosis"
                        elif "## 📋 plan" in low or "## plan" in low:
                            if current_section and buffer:
                                sections[current_section] = "\n".join(buffer).strip()
                                buffer = []
                            current_section = "plan"
                        elif "## 📚 resources" in low or "## resources" in low:
                            if current_section and buffer:
                                sections[current_section] = "\n".join(buffer).strip()
                                buffer = []
                            current_section = "resources"
                        elif "## 💡 coach" in low or "## coach" in low:
                            if current_section and buffer:
                                sections[current_section] = "\n".join(buffer).strip()
                                buffer = []
                            current_section = "insight"
                        elif current_section:
                            buffer.append(line)
                    
                    if current_section and buffer:
                        sections[current_section] = "\n".join(buffer).strip()
                    
                    # Render structured cards
                    if any(sections.values()):
                        if sections["diagnosis"]:
                            st.markdown(f"""
                            <div class='diagnosis-card'>
                                <div style='font-size:1rem; font-weight:700; color:#a78bfa; margin-bottom:0.6rem;'>🔍 Diagnosis — Learning Gaps Analysis</div>
                                <div style='color:#cbd5e1; font-size:0.9rem; line-height:1.7;'>{sections['diagnosis']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if sections["plan"]:
                            st.markdown(f"""
                            <div class='plan-card'>
                                <div style='font-size:1rem; font-weight:700; color:#4ade80; margin-bottom:0.6rem;'>📋 Personalized Study Plan</div>
                                <div style='color:#cbd5e1; font-size:0.9rem; line-height:1.7;'>{sections['plan']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if sections["resources"]:
                            st.markdown(f"""
                            <div class='resources-card'>
                                <div style='font-size:1rem; font-weight:700; color:#fbbf24; margin-bottom:0.6rem;'>📚 Resources & Tutorials</div>
                                <div style='color:#cbd5e1; font-size:0.9rem; line-height:1.7;'>{sections['resources']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if sections["insight"]:
                            st.markdown(f"""
                            <div class='insight-card'>
                                <div style='font-size:1rem; font-weight:700; color:#f472b6; margin-bottom:0.6rem;'>💡 Coach's Insight</div>
                                <div style='color:#cbd5e1; font-size:0.9rem; line-height:1.7;'>{sections['insight']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Store full response in session
                        st.session_state["session_steps"].append({
                            "prompt": prompt,
                            "reasoning": reasoning,
                            "response_preview": final_output[:200] + "..." if len(final_output) > 200 else final_output
                        })
                        st.session_state["messages"].append({"role": "assistant", "content": final_output})
                    else:
                        # Fallback: show raw markdown
                        st.markdown(final_output)
                        st.session_state["messages"].append({"role": "assistant", "content": final_output})

                except ValueError as ve:
                    st.error(f"⚠️ Configuration Error: {ve}\n\nPlease create a `.env` file with `GOOGLE_API_KEY=your_key_here`.")
                except Exception as e:
                    st.error(f"❌ Agent Execution Error: {type(e).__name__}: {e}")

    # ── CLEAR CHAT BUTTON
    if len(st.session_state["messages"]) > 1:
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state["messages"] = [st.session_state["messages"][0]]
            st.session_state["session_steps"] = []
            st.rerun()

# ─────────────────────────────────────────
# TAB 3: SESSION MEMORY PANEL
# ─────────────────────────────────────────
elif app_mode == "🧠 Session Memory":
    st.markdown("""
    <div class='main-header'>
        <h1>🧠 Session Memory & Progress Tracker</h1>
        <p>Track student learning interactions, reasoning chains, and session history across the AI Coach conversations.</p>
    </div>
    """, unsafe_allow_html=True)

    steps = st.session_state.get("session_steps", [])
    messages = st.session_state.get("messages", [])
    
    # ── KPIs
    user_msgs  = [m for m in messages if m["role"] == "user"]
    coach_msgs = [m for m in messages if m["role"] == "assistant"]
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(f"""<div class='metric-card'><div class='metric-value'>{len(user_msgs)}</div><div class='metric-label'>Student Queries</div></div>""", unsafe_allow_html=True)
    mc2.markdown(f"""<div class='metric-card'><div class='metric-value'>{len(steps)}</div><div class='metric-label'>Coach Sessions</div></div>""", unsafe_allow_html=True)
    mc3.markdown(f"""<div class='metric-card'><div class='metric-value'>{'Active' if steps else 'Empty'}</div><div class='metric-label'>Memory Status</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h3>📋 Session Interaction Log</h3></div>", unsafe_allow_html=True)
    
    if not steps:
        st.markdown("""
        <div style='background:#111827; border:1px dashed #334155; border-radius:12px; padding:2.5rem; text-align:center;'>
            <div style='font-size:2rem; margin-bottom:0.8rem;'>💾</div>
            <div style='color:#64748b; font-size:0.9rem;'>No coach sessions recorded yet.<br>Start a conversation in the <strong style='color:#60a5fa;'>🤖 AI Coach</strong> tab.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, step in enumerate(steps, 1):
            with st.expander(f"Session {i}: {step['prompt'][:60]}...", expanded=(i == len(steps))):
                st.markdown(f"**📝 Student Query:** {step['prompt']}")
                
                if step.get("reasoning"):
                    st.markdown("**🧠 Chain-of-Thought Reasoning:**")
                    for r in step["reasoning"]:
                        if r.strip():
                            st.markdown(f"""<div class='reasoning-step'>{r}</div>""", unsafe_allow_html=True)
                
                st.markdown(f"**🤖 Coach Response Preview:**")
                st.markdown(f"""<div class='memory-item'>{step['response_preview']}</div>""", unsafe_allow_html=True)

    # ── FULL CHAT LOG
    st.markdown("<div class='section-header'><h3>💬 Full Chat History</h3></div>", unsafe_allow_html=True)
    
    if len(messages) <= 1:
        st.info("No chat history yet. Start chatting with the AI Coach!", icon="ℹ️")
    else:
        for msg in messages[1:]:  # skip default greeting
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            role_label = "Student" if msg["role"] == "user" else "AI Coach"
            role_color = "#60a5fa" if msg["role"] == "user" else "#a78bfa"
            preview = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
            st.markdown(f"""
            <div class='memory-item' style='border-left-color:{role_color};'>
                <span style='font-weight:700; color:{role_color};'>{role_icon} {role_label}</span><br>
                <span style='color:#94a3b8;'>{preview}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── CLEAR MEMORY
    if steps or len(messages) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All Session Memory", type="secondary"):
            st.session_state["messages"] = [st.session_state["messages"][0]]
            st.session_state["session_steps"] = []
            st.rerun()
