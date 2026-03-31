import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Burnout Predictor",
    page_icon="🔥",
    layout="wide"
)

# ── Load model & artifacts ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = pickle.load(open("burnout_model.pkl",  "rb"))
    columns  = pickle.load(open("columns.pkl",        "rb"))
    encoder  = pickle.load(open("encoder.pkl",        "rb"))
    features = pickle.load(open("features.pkl",       "rb"))
    scaler   = pickle.load(open("std_scaler.pkl",     "rb"))
    return model, columns, encoder, features, scaler

try:
    model, columns, encoder, features, scaler = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔥 Employee Burnout Risk Predictor")
st.markdown("Fill in the employee details below to predict burnout risk.")

if not artifacts_loaded:
    st.error(f"❌ Could not load model files. Make sure all .pkl files are in the same folder as this app.\n\nError: {load_error}")
    st.stop()

st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This app uses a machine learning model trained on 850k employee records "
    "to predict burnout risk based on work habits, performance, and wellness metrics."
)

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("👤 Employee Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        role = st.selectbox("Role", [
            "Software Engineer", "Data Scientist", "Product Manager",
            "Designer", "HR", "Sales", "Marketing", "Finance", "Operations", "Other"
        ])
        job_level = st.selectbox("Job Level", ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"])
        department = st.selectbox("Department", [
            "Engineering", "Data", "Product", "Design", "HR",
            "Sales", "Marketing", "Finance", "Operations", "Other"
        ])
        tenure_months = st.slider("Tenure (months)", 1, 240, 24)
        salary = st.number_input("Salary (USD)", min_value=20000, max_value=500000, value=80000, step=1000)

    with col2:
        performance_score   = st.slider("Performance Score (1–10)",    1.0, 10.0, 7.0, 0.1)
        satisfaction_score  = st.slider("Satisfaction Score (1–10)",   1.0, 10.0, 6.0, 0.1)
        workload_score      = st.slider("Workload Score (1–10)",        1.0, 10.0, 7.0, 0.1)
        stress_level        = st.slider("Stress Level (1–10)",         1.0, 10.0, 6.0, 0.1)
        overtime_hours      = st.slider("Overtime Hours / week",        0,   40,   10)

    with col3:
        team_sentiment              = st.slider("Team Sentiment (1–10)",            1.0, 10.0, 6.0, 0.1)
        collaboration_score         = st.slider("Collaboration Score (1–10)",       1.0, 10.0, 7.0, 0.1)
        goal_achievement_rate       = st.slider("Goal Achievement Rate (0–1)",      0.0,  1.0, 0.75, 0.01)
        project_completion_rate     = st.slider("Project Completion Rate (0–1)",    0.0,  1.0, 0.80, 0.01)
        training_participation      = st.slider("Training Participation (0–1)",     0.0,  1.0, 0.50, 0.01)

    st.markdown("---")
    st.subheader("💬 Communication & Skills")

    col4, col5 = st.columns(2)
    with col4:
        recent_feedback        = st.selectbox("Recent Feedback",         ["Positive", "Neutral", "Negative"])
        communication_patterns = st.selectbox("Communication Patterns",  ["High", "Medium", "Low"])
        email_sentiment        = st.selectbox("Email Sentiment",         ["Positive", "Neutral", "Negative"])
        slack_activity         = st.selectbox("Slack Activity",          ["High", "Medium", "Low"])
        meeting_participation  = st.selectbox("Meeting Participation",   ["High", "Medium", "Low"])

    with col5:
        technical_skills          = st.slider("Technical Skills (1–10)",          1.0, 10.0, 7.0, 0.1)
        soft_skills               = st.slider("Soft Skills (1–10)",               1.0, 10.0, 7.0, 0.1)
        role_complexity_score     = st.slider("Role Complexity Score (1–10)",     1.0, 10.0, 6.0, 0.1)
        career_progression_score  = st.slider("Career Progression Score (1–10)", 1.0, 10.0, 6.0, 0.1)

    submitted = st.form_submit_button("🔍 Predict Burnout Risk", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Build raw input dict
    input_dict = {
        "tenure_months":             tenure_months,
        "salary":                    salary,
        "performance_score":         performance_score,
        "satisfaction_score":        satisfaction_score,
        "workload_score":            workload_score,
        "team_sentiment":            team_sentiment,
        "project_completion_rate":   project_completion_rate,
        "overtime_hours":            overtime_hours,
        "training_participation":    training_participation,
        "collaboration_score":       collaboration_score,
        "technical_skills":          technical_skills,
        "soft_skills":               soft_skills,
        "goal_achievement_rate":     goal_achievement_rate,
        "stress_level":              stress_level,
        "role_complexity_score":     role_complexity_score,
        "career_progression_score":  career_progression_score,
        # Categorical → encoded as simple ordinal mappings
        "recent_feedback":           {"Positive": 2, "Neutral": 1, "Negative": 0}[recent_feedback],
        "communication_patterns":    {"High": 2, "Medium": 1, "Low": 0}[communication_patterns],
        "email_sentiment":           {"Positive": 2, "Neutral": 1, "Negative": 0}[email_sentiment],
        "slack_activity":            {"High": 2, "Medium": 1, "Low": 0}[slack_activity],
        "meeting_participation":     {"High": 2, "Medium": 1, "Low": 0}[meeting_participation],
        "job_level":                 {"Junior": 0, "Mid": 1, "Senior": 2, "Lead": 3, "Manager": 4, "Director": 5}[job_level],
    }

    try:
        # Build DataFrame with exactly the columns the model expects
        input_df = pd.DataFrame([input_dict])

        # Add any missing columns as 0
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep only the model columns in the right order
        input_df = input_df[columns]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        risk_map = {
            "High":   ("🔴", "HIGH BURNOUT RISK",   "danger"),
            "Medium": ("🟡", "MEDIUM BURNOUT RISK", "warning"),
            "Low":    ("🟢", "LOW BURNOUT RISK",    "success"),
        }

        # Normalise prediction label
        pred_label = str(prediction).strip().capitalize()
        if pred_label not in risk_map:
            pred_label = "Medium"

        emoji, label, level = risk_map[pred_label]

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric("Burnout Risk Level", f"{emoji} {pred_label}")

        with col_r2:
            if proba is not None:
                classes = model.classes_
                for cls, prob in zip(classes, proba):
                    st.progress(float(prob), text=f"{cls}: {prob:.1%}")

        # Advice
        st.markdown("### 💡 Recommendations")
        if pred_label == "High":
            st.error(
                "⚠️ **Immediate attention needed.** Consider reducing workload, "
                "offering mental health support, flexible hours, and a 1:1 check-in with the manager."
            )
        elif pred_label == "Medium":
            st.warning(
                "🟡 **Monitor closely.** Encourage breaks, set realistic goals, "
                "and promote team collaboration activities."
            )
        else:
            st.success(
                "✅ **Employee is doing well.** Keep up the positive work environment "
                "and continue recognition programs."
            )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Make sure all .pkl files are in the same folder as burnout_app.py and match the trained model.")