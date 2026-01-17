import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sustainability Assistant",
    layout="wide",
    page_icon="🌱"
)

st.title("🌱 AI-Based Sustainability Intelligence Assistant")
st.markdown(
    "An **AI-powered system** for analyzing sustainability feedback, "
    "tracking trends across locations, and predicting escalation risks."
)

# ---------------- SESSION MEMORY ----------------
if "memory" not in st.session_state:
    st.session_state.memory = {"location": None, "time": None, "severity": None}

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- MONTHS ----------------
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# ---------------- ML MODEL ----------------
def train_resource_classifier():
    data = [
        ("power cut issue", "Electricity"),
        ("voltage fluctuation", "Electricity"),
        ("no water supply", "Water"),
        ("water leakage", "Water"),
        ("garbage not collected", "Waste"),
        ("plastic waste problem", "Waste"),
        ("air pollution", "Pollution"),
        ("vehicle smoke", "Pollution"),
        ("noise pollution", "Pollution")
    ]
    X, y = zip(*data)
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=300))
    ])
    model.fit(X, y)
    return model

ml_model = train_resource_classifier()

def classify_with_confidence(text):
    probs = ml_model.predict_proba([text])[0]
    classes = ml_model.classes_
    idx = np.argmax(probs)
    return classes[idx], round(probs[idx] * 100, 2)

# ---------------- SENTIMENT ----------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    return "Neutral"

# ---------------- CSV UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload Sustainability Feedback CSV (required column: feedback)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "feedback" not in df.columns:
        st.error("CSV must contain a 'feedback' column.")
        st.stop()

    df["Resource"], df["Confidence"] = zip(
        *df["feedback"].apply(classify_with_confidence)
    )
    df["Sentiment"] = df["feedback"].apply(get_sentiment)
    df["Month"] = np.random.choice(months, len(df))
    df["Location"] = np.random.choice(
        ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Delhi"],
        len(df)
    )
else:
    df = pd.DataFrame()

# ---------------- CHAT ASSISTANT ----------------
st.markdown("---")
st.subheader("💬 Sustainability AI Assistant")

user_input = st.chat_input("Describe your sustainability concern...")

def ai_reply():
    if st.session_state.memory["location"] is None:
        return "📍 Please specify your location (city or area)."
    if st.session_state.memory["severity"] is None:
        return "⚠️ How severe is the issue? (Low / Medium / High)"
    if st.session_state.memory["time"] is None:
        return "⏱️ When does this issue occur? (Daily / Weekly / Recently)"
    return "✅ Thanks! I will analyze the issue."

if user_input:
    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("assistant", ai_reply()))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

# ---------------- STORE MEMORY ----------------
for role, msg in st.session_state.chat[::-1]:
    if role == "user":
        if st.session_state.memory["location"] is None:
            st.session_state.memory["location"] = msg
        if st.session_state.memory["severity"] is None:
            if msg.lower() in ["low", "medium", "high"]:
                st.session_state.memory["severity"] = {"low":1,"medium":2,"high":3}[msg.lower()]
        if st.session_state.memory["time"] is None:
            if msg.lower() in ["daily","weekly","recently"]:
                st.session_state.memory["time"] = msg

# ---------------- FINAL AI ANALYSIS ----------------
if uploaded_file and user_input and all(st.session_state.memory.values()):
    resource, confidence = classify_with_confidence(user_input)
    sentiment = get_sentiment(user_input)
    st.markdown("### 🤖 AI Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Resource", resource)
    c2.metric("Confidence", f"{confidence}%")
    c3.metric("Sentiment", sentiment)

# ---------------- TREND ANALYZER (WITH LOCATION) ----------------
st.markdown("---")
st.subheader("📊 Sustainability Trend Analyzer (Location-wise)")

if uploaded_file and not df.empty:
    col1, col2 = st.columns(2)
    with col1:
        resource_filter = st.selectbox(
            "Select Resource", ["All"] + sorted(df["Resource"].unique())
        )
    with col2:
        location_filter = st.selectbox(
            "Select Location", ["All"] + sorted(df["Location"].unique())
        )

    trend_df = df.copy()
    if resource_filter != "All":
        trend_df = trend_df[trend_df["Resource"] == resource_filter]
    if location_filter != "All":
        trend_df = trend_df[trend_df["Location"] == location_filter]

    trend_df["Month"] = pd.Categorical(trend_df["Month"], categories=months, ordered=True)
    trend_df = trend_df.sort_values("Month")

    fig = px.bar(
        trend_df,
        x="Month",
        color="Location",
        title="📈 Sustainability Issues by Month and Location",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⬆️ Upload a CSV file to view trends.")

# ---------------- CITY HEATMAP ----------------
st.markdown("---")
st.subheader("🗺️ Sustainability Issue Heatmap by Location")

if uploaded_file and not df.empty:
    location_counts = df.groupby("Location").size().reset_index(name="Feedback Count")
    fig_map = px.choropleth(
        location_counts,
        locations="Location",
        locationmode="country names",
        color="Feedback Count",
        color_continuous_scale="Reds",
        title="🔥 Sustainability Issues by City"
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("⬆️ Upload a CSV file to view the location heatmap.")

# ---------------- DATA PREVIEW ----------------
with st.expander("📄 View Uploaded Dataset"):
    st.dataframe(df)
