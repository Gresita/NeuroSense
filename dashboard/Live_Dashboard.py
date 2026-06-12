import os
import time
import joblib
import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="NeuroSense Live Dashboard",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "neurosense_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_3class.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


bundle = load_model()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
label_names = bundle["label_names"]

df = load_data()

emotion_icons = {
    "Neutral": "😐",
    "Negative": "😢",
    "Positive": "😊"
}

st.title("🧠 NeuroSense Live Emotion Dashboard")
st.caption("Real-time emotion recognition using EEG and Eye Tracking data.")



st.markdown("---")


st.sidebar.title("Live Simulation Controls")

subjects = sorted(df["subject"].unique())
subject = st.sidebar.selectbox("Choose Subject", subjects)

sessions = sorted(df[df["subject"] == subject]["session"].unique())
session = st.sidebar.selectbox("Choose Session", sessions)

trials = sorted(df[(df["subject"] == subject) & (df["session"] == session)]["trial"].unique())
trial = st.sidebar.selectbox("Choose Trial", trials)

speed = st.sidebar.slider("Simulation Speed", 0.1, 2.0, 0.5)
run_simulation = st.sidebar.button("Start Simulation")

selected_df = df[
    (df["subject"] == subject) &
    (df["session"] == session) &
    (df["trial"] == trial)
].copy()

selected_df = selected_df.sort_values("sample")

if selected_df.empty:
    st.error("No data found for this selection.")
    st.stop()

sample_limit = st.sidebar.slider(
    "Number of Samples",
    5,
    min(80, len(selected_df)),
    min(30, len(selected_df))
)

selected_df = selected_df.head(sample_limit)

X_live = selected_df[feature_cols]

predictions = model.predict(X_live)
probabilities = model.predict_proba(X_live)

predicted_labels = [label_names[int(p)] for p in predictions]
confidences = probabilities.max(axis=1) * 100

current_label = predicted_labels[-1]
current_confidence = confidences[-1]
current_icon = emotion_icons.get(current_label, "🧠")

st.caption(
    f"Selected: Subject {subject} | Session {session} | Trial {trial} | Samples: {sample_limit}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Emotion",
        f"{current_icon} {current_label}",
        f"Confidence {current_confidence:.1f}%"
    )

with col2:
    st.metric("Subject", subject)

with col3:
    st.metric("Session", session)

with col4:
    st.metric("Trial", trial)

st.markdown("---")

timeline_df = pd.DataFrame({
    "Sample": selected_df["sample"].values,
    "Predicted Emotion": predicted_labels,
    "Confidence": confidences
})

left, right = st.columns([2, 1])

with left:
    st.subheader("Live Emotion Timeline")
    fig_timeline = px.line(
        timeline_df,
        x="Sample",
        y="Confidence",
        color="Predicted Emotion",
        markers=True
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with right:
    st.subheader("AI Insight")
    st.info(f"""
Current emotional state: {current_label}

Confidence: {current_confidence:.1f}%

The Random Forest model analyzed EEG and Eye Tracking features and detected patterns closest to the {current_label} emotional state.
""")

st.markdown("---")

col5, col6 = st.columns(2)

distribution = pd.Series(predicted_labels).value_counts().reset_index()
distribution.columns = ["Emotion", "Count"]

avg_proba = probabilities.mean(axis=0) * 100
confidence_df = pd.DataFrame({
    "Emotion": [label_names[int(c)] for c in model.classes_],
    "Confidence": avg_proba
})

with col5:
    st.subheader("Predicted Emotion Distribution")
    fig_pie = px.pie(
        distribution,
        names="Emotion",
        values="Count",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col6:
    st.subheader("Average Model Confidence")
    st.bar_chart(confidence_df.set_index("Emotion"))

st.markdown("---")

st.subheader("Top Influential Features")

rf_model = model.named_steps["rf"]
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(10)

fig_features = px.bar(
    feature_importance_df,
    x="Importance",
    y="Feature",
    orientation="h"
)
st.plotly_chart(fig_features, use_container_width=True)

st.markdown("---")

if run_simulation:
    status = st.empty()
    progress = st.progress(0)

    for i in range(sample_limit):
        status.success(
            f"Sample {i + 1}/{sample_limit} | "
            f"Prediction: {predicted_labels[i]} | "
            f"Confidence: {confidences[i]:.1f}%"
        )
        progress.progress((i + 1) / sample_limit)
        time.sleep(speed)

    st.success("Simulation completed successfully.")
else:
    st.warning("Choose subject/session/trial and click Start Simulation.")