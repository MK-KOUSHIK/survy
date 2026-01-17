import streamlit as st
import pandas as pd
import numpy as np
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
    "A **context-aware, conversational AI system** for sustainability issue analysis, "
    "trend comparison, forecasting, and escalation risk prediction."
)

# ---------------- SESSION MEMORY ----------------
if "memory" not in st.session_state:
    st.session_state.memory = {
        "location": None,
        "time": None,
        "severity": None
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
    return "Neutral"

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
        ["Bengaluru", "Chennai", "Hyderabad"],
        size=len(df)
    )

# ---------------- CHAT INTERFACE ----------------
st.markdown("---")
st.subheader("💬 Sustainability AI Assistant")

user_input = st.chat_input("Describe your sustainability concern...")

def ai_reply(text):
    t = text.lower()

    if not st.session_state.memory["location"] and not any(
        city in t for city in ["bengaluru", "chennai", "hyderabad"]
    ):
        return "📍 Please tell me your location (city)."

    if not st.session_state.memory["severity"] and not any(
        s in t for s in ["low", "medium", "high"]
    ):
        return "⚠️ How severe is the issue? (Low / Medium / High)"

    if not st.session_state.memory["time"] and not any(
        tm in t for tm in ["daily", "weekly", "recently"]
    ):
        return "⏱️ When does this issue occur? (Daily / Weekly / Recently)"

    return "✅ Thank you. I have enough information."

if user_input:
    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("assistant", ai_reply(user_input)))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

# ---------------- STORE MEMORY ----------------
for role, msg in st.session_state.chat[::-1]:
    if role == "user":
        m = msg.lower()
        if not st.session_state.memory["location"] and any(
            c in m for c in ["bengaluru", "chennai", "hyderabad"]
        ):
            st.session_state.memory["location"] = msg

        if not st.session_state.memory["severity"] and m in ["low","medium","high"]:
            st.session_state.memory["severity"] = {"low":1,"medium":2,"high":3}[m]

        if not st.session_state.memory["time"] and m in ["daily","weekly","recently"]:
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
    res = monthly_counts[monthly_counts["Resource"] == resource]
    if len(res) < 3:
        return None
    model = LinearRegression()
    model.fit(res[["MonthIndex"]], res["Count"])
    next_m = res["MonthIndex"].max() + 1
    return max(0, int(model.predict([[next_m]])[0]))

# ---------------- FINAL ANALYSIS ----------------
if all(st.session_state.memory.values()) and user_input:

    resource, confidence = classify_with_confidence(user_input)
    sentiment = get_sentiment(user_input)

    st.markdown("### 🤖 Final AI Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Resource", resource)
    c2.metric("Confidence", f"{confidence}%")
    c3.metric("Sentiment", sentiment)

    current_month = months[datetime.now().month - 1]
    last_month = months[datetime.now().month - 2]

    lm = len(df[(df["Month"] == last_month) & (df["Resource"] == resource)])
    cm = len(df[(df["Month"] == current_month) & (df["Resource"] == resource)])

    st.subheader("📊 Issue Comparison")
    st.bar_chart(
        pd.DataFrame(
            {"Feedback Count": [lm, cm]},
            index=[last_month, current_month]
        )
    )

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

    risk, score = escalation_risk(
        sentiment, cm - lm,
        st.session_state.memory["severity"],
        confidence
    )

    st.metric("🔮 Escalation Risk", risk, f"{score}/100")

    forecast = forecast_next_month(resource)
    if forecast:
        st.info(f"📈 Predicted complaints next month: **{forecast}**")

    st.subheader("🧠 AI Explanation")
    st.write(f"""
This analysis shows **{resource}** related sustainability issues.
Public sentiment is **{sentiment}**.
Escalation risk is **{risk}**, based on complaint trends, severity, and confidence.
Proactive intervention is recommended.
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
