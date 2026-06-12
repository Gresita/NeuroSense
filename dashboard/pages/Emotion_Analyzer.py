import os
import joblib
import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="🧪",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

st.title("Emotion Analyzer")
st.caption(
    "Enter values for the most influential EEG/Eye Tracking features and let the trained model predict the emotion."
)



st.markdown("---")

st.info(
    "The model was trained with 355 features. For simplicity, this page lets you modify only the most influential features. "
    "All other features are filled automatically using average values from the dataset."
)

rf_model = model.named_steps["rf"]
importances = rf_model.feature_importances_

top_features_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

baseline_input = df[feature_cols].mean().to_frame().T

st.subheader("Enter Feature Values")

edited_values = {}

for feature in top_features_df["Feature"]:
    min_value = float(df[feature].min())
    max_value = float(df[feature].max())
    mean_value = float(df[feature].mean())

    edited_values[feature] = st.slider(
        feature,
        min_value=min_value,
        max_value=max_value,
        value=mean_value
    )

for feature, value in edited_values.items():
    baseline_input.loc[0, feature] = value

st.markdown("---")

if st.button("Predict Emotion"):
    prediction = model.predict(baseline_input)[0]
    probabilities = model.predict_proba(baseline_input)[0]

    predicted_label = label_names[int(prediction)]
    confidence = probabilities.max() * 100
    icon = emotion_icons.get(predicted_label, "🧠")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Emotion", f"{icon} {predicted_label}")

    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")

    probability_df = pd.DataFrame({
        "Emotion": [label_names[int(c)] for c in model.classes_],
        "Probability": probabilities * 100
    })

    st.subheader("Prediction Probabilities")

    fig_prob = px.bar(
        probability_df,
        x="Emotion",
        y="Probability",
        text=probability_df["Probability"].round(1)
    )

    st.plotly_chart(fig_prob, use_container_width=True)

    st.subheader("AI Interpretation")
    st.success(
        f"The entered feature pattern was classified as {predicted_label} with {confidence:.1f}% confidence."
    )

else:
    st.warning("Enter values and click Predict Emotion.")

st.markdown("---")

st.info(
    "The prediction is generated using the trained Random Forest model based on EEG and Eye Tracking signals."
)