import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sustainability Dashboard",
    layout="wide",
    page_icon="🌱"
)

# ---------------- DARK THEME ----------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------- TITLE ----------------
st.title("🌱 AI-Based Sustainability Dashboard")
st.caption(
    "Automatically analyzes ANY CSV file and generates "
    "AI-powered sustainability insights."
)

# ---------------- CONSTANTS ----------------
MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

STATES = [
    "Tamil Nadu","Karnataka","Kerala","Maharashtra",
    "Delhi","Telangana","Andhra Pradesh"
]

RESOURCE_KEYWORDS = {
    "Electricity": ["electric", "power", "voltage", "current"],
    "Water": ["water", "leak", "pipeline", "sewage"],
    "Waste": ["waste", "garbage", "plastic", "trash"],
    "Pollution": ["pollution", "smoke", "air", "noise"]
}

# ---------------- CSV UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload ANY CSV file", type=["csv"])

if not uploaded_file:
    st.info("⬆️ Upload a CSV file to begin analysis.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# ---------------- COLUMN DETECTION ----------------
def find_column(keywords):
    for col in df_raw.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

text_col = find_column(["feedback", "comment", "description", "issue", "text"])
state_col = find_column(["state", "location", "city", "region"])
month_col = find_column(["month"])
date_col = find_column(["date", "time"])
value_col = find_column(["usage", "count", "value", "number"])

# ---------------- NORMALIZE DATA ----------------
df = pd.DataFrame()

df["Text"] = (
    df_raw[text_col].astype(str)
    if text_col else "No description"
)

df["State"] = (
    df_raw[state_col]
    if state_col else np.random.choice(STATES, len(df_raw))
)

if month_col:
    df["Month"] = df_raw[month_col]
elif date_col:
    df["Month"] = pd.to_datetime(
        df_raw[date_col], errors="coerce"
    ).dt.month_name()
else:
    df["Month"] = np.random.choice(MONTHS, len(df_raw))

df["Usage"] = (
    pd.to_numeric(df_raw[value_col], errors="coerce").fillna(0)
    if value_col else np.random.randint(40, 90, len(df_raw))
)

# ---------------- RESOURCE CLASSIFICATION ----------------
def classify_resource(text):
    text = text.lower()
    scores = {
        r: sum(k in text for k in ks)
        for r, ks in RESOURCE_KEYWORDS.items()
    }
    return max(scores, key=scores.get)

df["Resource"] = df["Text"].apply(classify_resource)

# ---------------- MONTH ORDER ----------------
df["Month"] = pd.Categorical(df["Month"], categories=MONTHS, ordered=True)
df = df.sort_values("Month")

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🔎 Filters")

selected_states = st.sidebar.multiselect(
    "State",
    sorted(df["State"].unique()),
    default=df["State"].unique()
)

selected_resources = st.sidebar.multiselect(
    "Resource",
    sorted(df["Resource"].unique()),
    default=df["Resource"].unique()
)

filtered_df = df[
    (df["State"].isin(selected_states)) &
    (df["Resource"].isin(selected_resources))
]

# ---------------- TREND GRAPH ----------------
st.subheader("📈 Resource Usage Trends")

fig = px.line(
    filtered_df,
    x="Month",
    y="Usage",
    color="Resource",
    line_group="State",
    markers=True,
    template="plotly_dark"
)

fig.update_layout(
    hovermode="x unified",
    height=520,
    xaxis_title="Month",
    yaxis_title="Usage Level"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- SPIKE ALERTS ----------------
st.markdown("---")
st.subheader("🚨 Anomaly Alerts")

for (state, resource), g in filtered_df.groupby(["State","Resource"]):
    if g["Usage"].mean() > 0 and g["Usage"].max() > g["Usage"].mean() * 1.3:
        st.error(f"⚠️ Spike detected in {state} – {resource}")

# ---------------- OPENAI INSIGHT FUNCTION ----------------
def generate_ai_insight(df):
    summary = (
        df.groupby("State")["Usage"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    prompt = f"""
You are a sustainability policy analyst.

Here is average sustainability stress by Indian states:
{summary.to_string()}

Tasks:
1. Identify key trends
2. Highlight high-risk states
3. Suggest government or NGO actions
4. Keep explanation short and clear
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert sustainability analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content

# ---------------- AI INSIGHTS ----------------
st.markdown("---")
st.subheader("🤖 AI-Generated Insights")

if st.button("Generate AI Explanation"):
    with st.spinner("Analyzing data with AI..."):
        ai_text = generate_ai_insight(df)
        st.success(ai_text)

# ---------------- DATA VIEW ----------------
with st.expander("📊 View Normalized Data"):
    st.dataframe(df)
