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
    "A **context-aware, conversational AI system** for analyzing sustainability feedback "
    "with trend comparison and escalation risk prediction."
)

# ---------------- SESSION MEMORY ----------------
if "memory" not in st.session_state:
    st.session_state.memory = {
        "location": None,
        "time": None,
        "severity": None,
        "specific_issue": None
    }

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- MONTHS ----------------
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# ---------------- ML MODEL ----------------
def train_resource_classifier():
    train_data = [
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
    X, y = zip(*train_data)
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=300))
    ])
    model.fit(X, y)
    return model

ml_model = train_resource_classifier()

def classify_with_confidence(text):
    probs = ml_model.predict_proba([str(text)])[0]
    classes = ml_model.classes_
    idx = np.argmax(probs)
    return classes[idx], round(probs[idx] * 100, 2)

# ---------------- SENTIMENT ----------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# ---------------- CONTEXT DETECTION ----------------
def detect_missing_context(text):
    text = text.lower()
    missing = []

    if any(p in text for p in ["my area", "near me", "here", "local"]):
        missing.append("location")

    if any(p in text for p in ["often", "recently", "nowadays"]):
        missing.append("time")

    if len(text.split()) < 5:
        missing.append("specific_issue")

    if not any(p in text for p in ["daily", "frequent", "severe", "minor"]):
        missing.append("severity")

    return missing

# ---------------- DATASET UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Sustainability Feedback Dataset (CSV format) — required column: `feedback`",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "feedback" not in df.columns:
        st.error("Dataset must contain a `feedback` column.")
        st.stop()
else:
    df = pd.DataFrame(columns=["feedback"])

if not df.empty:
    df["Resource"], df["Confidence"] = zip(*df["feedback"].apply(classify_with_confidence))
    df["Sentiment"] = df["feedback"].apply(get_sentiment)
    df["Month"] = np.random.choice(months, size=len(df))
    df["Location"] = np.random.choice(
        ["Bengaluru", "Whitefield", "Indiranagar", "Chennai", "Hyderabad"],
        size=len(df)
    )

# ---------------- CHAT INTERFACE ----------------
st.markdown("---")
st.subheader("💬 Sustainability AI Assistant")

user_input = st.chat_input("Describe your sustainability concern...")

def ai_reply(user_text):
    missing = detect_missing_context(user_text)

    if "location" in missing and not st.session_state.memory["location"]:
        return "📍 Could you please specify your **location (city / locality)**?"

    if "severity" in missing and not st.session_state.memory["severity"]:
        return "⚖️ How severe is the issue? (Low / Medium / High)"

    if "time" in missing and not st.session_state.memory["time"]:
        return "⏱️ When does this issue usually occur? (Daily / Weekly / Recently)"

    return "✅ Thank you. I have enough information to analyze this issue."

if user_input:
    st.session_state.chat.append(("user", user_input))
    reply = ai_reply(user_input)
    st.session_state.chat.append(("assistant", reply))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

# ---------------- STORE MEMORY FROM FOLLOW-UPS ----------------
for role, msg in st.session_state.chat[::-1]:
    if role == "user":
        if st.session_state.memory["location"] is None and any(
            city.lower() in msg.lower() for city in ["bengaluru","chennai","hyderabad"]
        ):
            st.session_state.memory["location"] = msg

        if st.session_state.memory["severity"] is None and msg.lower() in ["low","medium","high"]:
            st.session_state.memory["severity"] = {"low":1,"medium":2,"high":3}[msg.lower()]

        if st.session_state.memory["time"] is None and msg.lower() in ["daily","weekly","recently"]:
            st.session_state.memory["time"] = msg

# ---------------- FINAL ANALYSIS ----------------
if all(st.session_state.memory.values()) and user_input:

    predicted_resource, confidence = classify_with_confidence(user_input)
    sentiment = get_sentiment(user_input)

    st.markdown("### 🤖 Final AI Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Resource", predicted_resource)
    c2.metric("Confidence", f"{confidence}%")
    c3.metric("Sentiment", sentiment)

    # -------- Last Month vs Now --------
    current_month = months[datetime.now().month - 1]
    last_month = months[datetime.now().month - 2]

    lm = len(df[(df["Month"]==last_month) & (df["Resource"]==predicted_resource)])
    cm = len(df[(df["Month"]==current_month) & (df["Resource"]==predicted_resource)])

    cmp_df = pd.DataFrame({
        "Period":[last_month,current_month],
        "Feedback Count":[lm,cm]
    })

    fig_cmp = px.bar(
        cmp_df,
        x="Period",
        y="Feedback Count",
        text="Feedback Count",
        title=f"{predicted_resource} Issues: Last Month vs Current Month"
    )

    st.plotly_chart(fig_cmp, use_container_width=True)

    # -------- Escalation Risk --------
    def escalation_risk(sentiment, trend_change, severity, confidence):
        score = 0
        if sentiment == "Negative":
            score += 30
        if trend_change > 0:
            score += 25
        score += severity * 10
        score += confidence * 0.2

        if score >= 70:
            return "High", score
        elif score >= 40:
            return "Medium", score
        else:
            return "Low", score

    risk, score = escalation_risk(
        sentiment,
        cm - lm,
        st.session_state.memory["severity"],
        confidence
    )

    st.metric("🔮 Escalation Risk", risk, f"{int(score)}/100")

# ---------------- DATA PREVIEW ----------------
with st.expander("📄 View Dataset"):
    st.dataframe(df)
