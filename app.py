import streamlit as st
import pandas as pd
import numpy as np
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
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.title("🌱 AI-Based Sustainability Dashboard")
st.caption(
    "Analyzing large-scale sustainability feedback data to detect "
    "resource usage trends across states and seasons."
)

# ---------------- SAMPLE DATA (REPLACE WITH CSV LATER IF NEEDED) ----------------
months = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

states = ["Tamil Nadu", "Karnataka", "Kerala"]
resources = ["Electricity", "Water", "Waste"]

data = []
np.random.seed(42)

for state in states:
    for resource in resources:
        for month in months:
            data.append({
                "State": state,
                "Resource": resource,
                "Month": month,
                "Usage": np.random.randint(40, 90)
            })

df = pd.DataFrame(data)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🔎 Filter Data")

selected_states = st.sidebar.multiselect(
    "Select State(s)",
    options=df["State"].unique(),
    default=df["State"].unique()
)

selected_resources = st.sidebar.multiselect(
    "Select Resource(s)",
    options=df["Resource"].unique(),
    default=df["Resource"].unique()
)

# ---------------- APPLY FILTERS ----------------
filtered_df = df[
    (df["State"].isin(selected_states)) &
    (df["Resource"].isin(selected_resources))
]

# Ensure correct month order
filtered_df["Month"] = pd.Categorical(
    filtered_df["Month"],
    categories=months,
    ordered=True
)
filtered_df = filtered_df.sort_values("Month")

# ---------------- MAIN GRAPH ----------------
st.subheader("📈 Resource Usage Trends (Hover to Explore Details)")

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
    legend_title="Resource Type",
    hovermode="x unified",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- DATA VIEW ----------------
with st.expander("📊 View Processed Data"):
    st.dataframe(filtered_df, use_container_width=True)
