import streamlit as st
import pandas as pd
from collections import Counter
import re

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Sustainability Issue Trend Analyzer", layout="wide")

st.title("🌱 AI-Based Sustainability Issue Trend Analyzer")
st.write("Analyze community feedback to identify sustainability issue trends.")

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -------------------- KEYWORD-BASED ANALYSIS --------------------
def extract_keywords(texts):
    all_words = []
    for text in texts:
        cleaned = clean_text(text)
        words = cleaned.split()
        all_words.extend(words)

    stopwords = {
        "the", "is", "and", "to", "of", "in", "for", "on", "with",
        "a", "an", "this", "that", "it", "are", "was", "were"
    }

    filtered_words = [w for w in all_words if w not in stopwords and len(w) > 3]
    return Counter(filtered_words).most_common(10)

# -------------------- USER INPUT --------------------
st.subheader("📥 Enter Community Feedback")

user_input = st.text_area(
    "Paste sustainability-related feedback (one sentence per line):",
    height=200
)

if st.button("Analyze Feedback"):

    if not user_input.strip():
        st.warning("Please enter some feedback text.")
    else:
        feedback_list = user_input.split("\n")

        # Keyword extraction
        keywords = extract_keywords(feedback_list)

        if not keywords:
            st.error("Not enough meaningful data to analyze.")
        else:
            df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])

            # -------------------- RESULTS --------------------
            st.subheader("📊 Key Sustainability Issues")
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Issue Frequency Chart")
            st.bar_chart(df.set_index("Keyword"))

            # -------------------- INSIGHT --------------------
            top_issue = df.iloc[0]["Keyword"]

            st.success(f"🔍 **Most Reported Sustainability Issue:** `{top_issue}`")

            st.info(
                "💡 **Insight:** Local authorities can prioritize this issue "
                "for faster policy action and resource allocation."
            )

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built using Streamlit | AI Survy Sustainability Challenge")
