import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sustainability Assistant",
    layout="wide",
    page_icon="🌱"
)

st.title("🌱 AI-Based Sustainability Intelligence Assistant")
st.markdown(
    "A **context-aware, conversational AI system** for analyzing sustainability feedback, "
    "trend comparison, forecasting, and escalation risk prediction."
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

# ---------------- ML CLASSIFIER ----------------
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

    if "location" not in st.session_state.memory and not any(
        city in text for city in ["bengaluru", "chennai", "hyderabad"]
    ):
        missing.append("location")

    if not any(w in text for w in ["daily", "weekly", "recently"]):
        missing.append("time")

    if not any(w in text for w in ["low", "medium", "high"]):
        missing.append("severity")

    return missing

# ---------------- DATASET UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Sustainability Feedback Dataset (CSV with `feedback` column)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "feedback" not in df.columns:
        st.error("CSV must contain a `feedback` column")
        st.stop()
else:
    df = pd.DataFrame(columns=["feedback"])

# ---------------- DATA PROCESSING ----------------
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

def ai_reply(text):
    missing = detect_missing_context(text)

    if "location" in missing and not st.session_state.memory["location"]:
        return "📍 Please specify your location (city / area)."

    if "severity" in missing and not st.session_state.memory["severity"]:
        return "⚠️ How severe is the issue? (Low / Medium / High)"

    if "time" in missing and not st.session_state.memory["time"]:
        return "⏱️ When does this issue occur? (Daily / Weekly / Recently)"

    return "✅ Thanks! I have enough information to analyze this issue."

if user_input:
    st.session_state.chat.append(("user", user_input))
    reply = ai_reply(user_input)
    st.session_state.chat.append(("assistant", reply))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

# ---------------- STORE MEMORY ----------------
for role, msg in st.session_state.chat[::-1]:
    if role == "user":
        if not st.session_state.memory["location"] and any(
            c in msg.lower() for c in ["bengaluru", "chennai", "hyderabad"]
        ):
            st.session_state.memory["location"] = msg

        if not st.session_state.memory["severity"] and msg.lower() in ["low","medium","high"]:
            st.session_state.memory["severity"] = {"low":1,"medium":2,"high":3}[msg.lower()]

        if not st.session_state.memory["time"] and msg.lower() in ["daily","weekly","recently"]:
            st.session_state.memory["time"] = msg

# ---------------- FORECASTING ----------------
month_index = {m:i for i,m in enumerate(months)}
if not df.empty:
    df["MonthIndex"] = df["Month"].map(month_index)

monthly_counts = (
    df.groupby(["Resource","MonthIndex"])
    .size()
    .reset_index(name="Count")
)

def forecast_next_month(resource):
    res_df = monthly_counts[monthly_counts["Resource"] == resource]
    if len(res_df) < 3:
        return None

    X = res_df[["MonthIndex"]]
    y = res_df["Count"]
    model = LinearRegression()
    model.fit(X, y)

    next_month = X["MonthIndex"].max() + 1
    return max(0, int(model.predict([[next_month]])[0]))

# ---------------- FINAL ANALYSIS ----------------
if all(st.session_state.memory.values()) and user_input:
    predicted_resource, confidence = classify_with_confidence(user_input)
    sentiment = get_sentiment(user_input)

    st.markdown("### 🤖 Final AI Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Resource", predicted_resource)
    c2.metric("Confidence", f"{confidence}%")
    c3.metric("Sentiment", sentiment)

    current_month = months[datetime.now().month - 1]
    last_month = months[datetime.now().month - 2]

    lm = len(df[(df["Month"]==last_month) & (df["Resource"]==predicted_resource)])
    cm = len(df[(df["Month"]==current_month) & (df["Resource"]==predicted_resource)])

    fig = px.bar(
        pd.DataFrame({
            "Period":[last_month,current_month],
            "Count":[lm,cm]
        }),
        x="Period",
        y="Count",
        title="📊 Last Month vs Current Month"
    )
    st.plotly_chart(fig, use_container_width=True)

    def escalation_risk(sentiment, trend, severity, confidence):
        score = 0
        if sentiment == "Negative":
            score += 30
        if trend > 0:
            score += 25
        score += severity * 10
        score += confidence * 0.2

        if score >= 70:
            return "High", int(score)
        elif score >= 40:
            return "Medium", int(score)
        return "Low", int(score)

    risk, score = escalation_risk(sentiment, cm-lm,
                                  st.session_state.memory["severity"],
                                  confidence)

    st.metric("🔮 Escalation Risk", risk, f"{score}/100")

    forecast = forecast_next_month(predicted_resource)
    if forecast is not None:
        st.info(f"📈 Predicted complaints next month: **{forecast}**")

    st.subheader("🧠 AI Explanation")
    st.write(f"""
The system analyzed citizen feedback related to **{predicted_resource}**.
Sentiment is **{sentiment}**, indicating public perception.
Escalation risk is **{risk}**, based on trend growth, severity, and confidence.
If no action is taken, complaints are likely to continue.
""")

# ---------------- WORD CLOUD ----------------
if not df.empty:
    st.subheader("☁️ Public Feedback Word Cloud")
    text = " ".join(df["feedback"].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ---------------- DATA PREVIEW ----------------
with st.expander("📄 View Dataset"):
    st.dataframe(df)
