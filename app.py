import streamlit as st
import pandas as pd
import numpy as np
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sustainability Trend Analyzer",
    layout="wide",
    page_icon="🌱"
)

st.title("🌱 Sustainability Issue Trend Analyzer")
st.write(
    "A lightweight AI-inspired system for analyzing sustainability feedback, "
    "tracking trends across **locations and months**, and generating alerts."
)

# ---------------- MONTHS ----------------
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# ---------------- ISSUE CLASSIFICATION (KEYWORD AI) ----------------
RESOURCE_KEYWORDS = {
    "Water": ["water", "leak", "pipeline", "tap", "sewage"],
    "Electricity": ["power", "electric", "voltage", "current", "cut"],
    "Waste": ["garbage", "waste", "plastic", "trash", "dump"],
    "Pollution": ["pollution", "smoke", "noise", "air", "burning"]
}

def classify_issue(text):
    text = text.lower()
    scores = {k: sum(word in text for word in v)
              for k, v in RESOURCE_KEYWORDS.items()}
    issue = max(scores, key=scores.get)
    confidence = min(95, scores[issue] * 25)
    return issue, confidence

# ---------------- SIMPLE SENTIMENT ----------------
NEGATIVE_WORDS = ["bad", "problem", "issue", "dirty", "polluted", "leak", "cut", "delay"]
POSITIVE_WORDS = ["good", "clean", "fixed", "improved", "resolved"]

def get_sentiment(text):
    t = text.lower()
    neg = sum(w in t for w in NEGATIVE_WORDS)
    pos = sum(w in t for w in POSITIVE_WORDS)
    if neg > pos:
        return "Negative"
    elif pos > neg:
        return "Positive"
    return "Neutral"

# ---------------- CSV UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload Sustainability Feedback CSV (required column: feedback)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "feedback" not in df.columns:
        st.error("CSV must contain a column named `feedback`.")
        st.stop()

    # AI-style analysis
    df["Issue"], df["Confidence"] = zip(
        *df["feedback"].apply(classify_issue)
    )
    df["Sentiment"] = df["feedback"].apply(get_sentiment)

    # Demo metadata
    df["Month"] = np.random.choice(months, len(df))
    df["Location"] = np.random.choice(
        ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Delhi"],
        len(df)
    )

    # Severity score
    df["Severity"] = df["Sentiment"].map({
        "Negative": 3,
        "Neutral": 2,
        "Positive": 1
    })

else:
    df = pd.DataFrame()

# ---------------- TREND ANALYZER ----------------
st.markdown("---")
st.subheader("📊 Trend Analyzer (Month × Location)")

if not df.empty:

    chart_df = (
        df.groupby(["Month", "Location"])
        .size()
        .reset_index(name="Feedback Count")
    )

    chart_df["Month"] = pd.Categorical(
        chart_df["Month"], categories=months, ordered=True
    )
    chart_df = chart_df.sort_values("Month")

    st.bar_chart(
        chart_df,
        x="Month",
        y="Feedback Count",
        color="Location"
    )

else:
    st.info("⬆️ Upload a CSV file to view trends.")

# ---------------- LOCATION ALERTS ----------------
st.markdown("---")
st.subheader("🚨 Location-wise Alerts")

if not df.empty:
    alerts = df.groupby("Location").size().reset_index(name="Issues")

    for _, row in alerts.iterrows():
        if row["Issues"] >= 4:
            st.error(f"High number of issues reported in **{row['Location']}**")
        else:
            st.success(f"{row['Location']} is under normal range")

# ---------------- INSIGHTS ----------------
st.markdown("---")
st.subheader("🧠 AI Insights")

if not df.empty:
    top_issue = df["Issue"].value_counts().idxmax()
    top_location = df["Location"].value_counts().idxmax()

    st.info(
        f"Most reported issue: **{top_issue}**\n\n"
        f"Most affected location: **{top_location}**\n\n"
        "Recommended action: Prioritize inspections and preventive measures."
    )

# ---------------- CSV EXPORT ----------------
st.markdown("---")
st.subheader("📥 Download Analyzed Data")

if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Analysis CSV",
        csv,
        file_name="sustainability_analysis.csv",
        mime="text/csv"
    )

# ---------------- DATA PREVIEW ----------------
with st.expander("📄 View Full Dataset"):
    st.dataframe(df)
