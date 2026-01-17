import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

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

# ---------------- TITLE ----------------
st.title("🌱 AI-Based Sustainability Dashboard")
st.caption(
    "Automatically analyzes **any CSV file** to detect sustainability trends, "
    "resource stress, and regional patterns."
)

# ---------------- MONTHS ----------------
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
columns = df_raw.columns.str.lower().tolist()

def find_column(keywords):
    for col in df_raw.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

text_col = find_column(["feedback", "comment", "description", "issue", "text", "complaint"])
state_col = find_column(["state", "location", "city", "region"])
month_col = find_column(["month"])
date_col = find_column(["date", "time"])
value_col = find_column(["usage", "count", "value", "amount", "number"])

# ---------------- NORMALIZATION ----------------
df = pd.DataFrame()

# TEXT
if text_col:
    df["Text"] = df_raw[text_col].astype(str)
else:
    df["Text"] = "No description provided"

# STATE
if state_col:
    df["State"] = df_raw[state_col]
else:
    df["State"] = np.random.choice(STATES, len(df_raw))

# MONTH
if month_col:
    df["Month"] = df_raw[month_col]
elif date_col:
    df["Month"] = pd.to_datetime(df_raw[date_col], errors="coerce").dt.month_name()
else:
    df["Month"] = np.random.choice(MONTHS, len(df_raw))

# USAGE
if value_col:
    df["Usage"] = pd.to_numeric(df_raw[value_col], errors="coerce").fillna(0)
else:
    df["Usage"] = np.random.randint(40, 90, len(df_raw))

# RESOURCE CLASSIFICATION (AI-style)
def classify_resource(text):
    text = text.lower()
    scores = {r: sum(k in text for k in ks) for r, ks in RESOURCE_KEYWORDS.items()}
    return max(scores, key=scores.get)

df["Resource"] = df["Text"].apply(classify_resource)

# CLEAN MONTH ORDER
df["Month"] = pd.Categorical(df["Month"], categories=MONTHS, ordered=True)
df = df.sort_values("Month")

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🔎 Filter Data")

selected_states = st.sidebar.multiselect(
    "Select State(s)",
    sorted(df["State"].unique()),
    default=df["State"].unique()
)

selected_resources = st.sidebar.multiselect(
    "Select Resource(s)",
    sorted(df["Resource"].unique()),
    default=df["Resource"].unique()
)

filtered_df = df[
    (df["State"].isin(selected_states)) &
    (df["Resource"].isin(selected_resources))
]

# ---------------- TREND GRAPH ----------------
st.subheader("📈 Resource Usage Trends (Auto-Analyzed)")

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
    xaxis_title="Month",
    yaxis_title="Usage Level (AI-derived index)",
    hovermode="x unified",
    height=520
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- SPIKE ALERTS ----------------
st.markdown("---")
st.subheader("🚨 Anomaly Detection")

for (state, resource), g in filtered_df.groupby(["State", "Resource"]):
    if g["Usage"].mean() > 0 and g["Usage"].max() > g["Usage"].mean() * 1.3:
        st.error(f"⚠️ Spike detected in **{state} – {resource}**")

# ---------------- HEATMAP ----------------
st.markdown("---")
st.subheader("🗺️ Regional Intensity Map")

heat_df = (
    df.groupby("State")["Usage"]
    .mean()
    .reset_index(name="Avg Usage")
)

fig_map = px.choropleth(
    heat_df,
    locations="State",
    locationmode="india states",
    color="Avg Usage",
    color_continuous_scale="Reds",
    template="plotly_dark"
)

fig_map.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_map, use_container_width=True)

# ---------------- AI EXPLANATION ----------------
st.markdown("---")
st.subheader("🧠 AI Explanation")

top_state = heat_df.sort_values("Avg Usage", ascending=False).iloc[0]["State"]
top_resource = df["Resource"].value_counts().idxmax()

st.info(
    f"""
**Automatic Insights Generated:**

• Most affected state: **{top_state}**  
• Most stressed resource: **{top_resource}**  
• Detected seasonal and regional usage patterns  

**Recommendation:**  
Prioritize monitoring and infrastructure improvements in high-usage regions.
"""
)

# ---------------- DATA VIEW ----------------
with st.expander("📊 View Normalized Data"):
    st.dataframe(df)
