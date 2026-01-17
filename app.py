import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
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
    "An **AI-inspired conversational system** for sustainability issue analysis, "
    "trend comparison, forecasting, and escalation risk estimation."
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

# ---------------- RULE-BASED ISSUE CLASSIFIER ----------------
RESOURCE_KEYWORDS = {
    "Water": ["water", "leak", "pipeline", "supply"],
    "Electricity": ["power", "voltage", "current", "cut"],
    "Waste": ["garbage", "waste", "plastic", "trash"],
    "Pollution": ["pollution", "smoke", "noise", "air"]
}

def classify_resource(text):
    text = text.lower()
    scores = {}
    for resource, words in RESOURCE_KEYWORDS.items():
        scores[resource] = sum(w in text for w in words)

    resource = max(scores, key=scores.get)
    confidence = scores[resource] * 25
    return resource, min(confidence, 95)

# ---------------- SIMPLE SENTIMENT ----------------
NEGATIVE = ["bad","problem","issue","no","not","dirty","polluted","leak","cut","delay"]
POSITIVE = ["good","clean","better","fixed","improved"]

def get_sentiment(text):
    t = text.lower()
    neg = sum(w in t for w in NEGATIVE)
    pos = sum(w in t for w in POSITIVE)

    if neg > pos:
        return "Negative"
    elif pos > neg:
        return "Positive"
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
    df["Resource"], df["Confidence"] = zip(*df["feedback"].apply(classify_resource))
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
        c in t for c in ["bengaluru", "chennai", "hyderabad"]
    ):
        return "📍 Please specify your location (city)."

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

# ---------------- FORECASTING (RULE-BASED) ----------------
def forecast_next_month(resource):
    if df.empty:
        return None
    counts = df[df["Resource"] == resource].groupby("Month").size()
    if len(counts) < 2:
        return None
    return int(counts.mean() * 1.1)

# ---------------- FINAL ANALYSIS ----------------
if all(st.session_state.memory.values()) and user_input:

    resource, confidence = classify_resource(user_input)
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

    score = (confidence * 0.4) + (st.session_state.memory["severity"] * 20)
    risk = "High" if score > 70 else "Medium" if score > 40 else "Low"

    st.metric("🔮 Escalation Risk", risk, f"{int(score)}/100")

    forecast = forecast_next_month(resource)
    if forecast:
        st.info(f"📈 Estimated complaints next month: **{forecast}**")

    st.subheader("🧠 AI Explanation")
    st.write(
        f"The system detected **{resource}** issues with **{sentiment}** sentiment. "
        f"Risk level is **{risk}**, based on complaint patterns and severity. "
        "Preventive action is recommended."
    )

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
