import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="📊 Universal Forms AI Analyzer",
    layout="wide"
)

# ==================================================
# SMART COLUMN DETECTION
# ==================================================
def detect_score_column(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None

    keywords = ['score', 'rate', 'rating', 'satisf', 'mark', 'grade']
    for col in numeric_cols:
        if any(k in col.lower() for k in keywords):
            return col

    return df[numeric_cols].var().idxmax()


def detect_group_column(df):
    for col in df.select_dtypes(include='object').columns:
        if any(x in col.lower() for x in ['group', 'dept', 'team', 'class', 'division']):
            return col
    return None


def detect_feedback_column(df):
    for col in df.select_dtypes(include='object').columns:
        if any(x in col.lower() for x in ['feedback', 'comment', 'remark', 'review', 'note']):
            return col
    return None

# ==================================================
# AI CHART ENGINE
# ==================================================
def ai_chart_engine(df, score_col, group_col):
    charts = []

    # Group comparison
    if score_col and group_col:
        bar = alt.Chart(df).mark_bar().encode(
            x=alt.X(group_col, sort='-y', title=group_col),
            y=alt.Y(score_col, aggregate='mean', title='Average Score'),
            color=alt.Color(score_col, aggregate='mean', scale=alt.Scale(scheme='blues')),
            tooltip=[
                group_col,
                alt.Tooltip(score_col, aggregate='mean', title='Avg Score')
            ]
        ).properties(
            title="🏢 Average Score by Group"
        )
        charts.append(bar)

    # Distribution
    if score_col:
        hist = alt.Chart(df).mark_bar(opacity=0.8).encode(
            x=alt.X(score_col, bin=alt.Bin(maxbins=20), title="Score Range"),
            y=alt.Y('count()', title="Responses"),
            tooltip=['count()']
        ).properties(
            title="📊 Score Distribution"
        )
        charts.append(hist)

    # Outlier detection
    if score_col:
        box = alt.Chart(df).mark_boxplot(extent='min-max').encode(
            y=alt.Y(score_col, title="Score")
        ).properties(
            title="🚨 Outlier Detection"
        )
        charts.append(box)

    return charts

# ==================================================
# CORRELATION HEATMAP
# ==================================================
def ai_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr().reset_index().melt('index')
    corr.columns = ['X', 'Y', 'Correlation']

    heatmap = alt.Chart(corr).mark_rect().encode(
        x='X:O',
        y='Y:O',
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue')),
        tooltip=['X', 'Y', 'Correlation']
    ).properties(
        title="🧠 Correlation Heatmap"
    )

    return heatmap

# ==================================================
# NATURAL LANGUAGE AI EXPLANATION
# ==================================================
def ai_explain_insights(df, score_col, group_col):
    if not score_col:
        return "⚠️ No numeric score column detected for explanation."

    explanation = []

    mean = df[score_col].mean()
    std = df[score_col].std()

    explanation.append(
        f"The overall average **{score_col}** is **{mean:.2f}**. "
        f"The data shows {'high' if std > mean * 0.3 else 'low'} variability, "
        f"indicating {'inconsistent' if std > mean * 0.3 else 'stable'} performance."
    )

    if group_col:
        group_means = df.groupby(group_col)[score_col].mean()
        best = group_means.idxmax()
        worst = group_means.idxmin()

        explanation.append(
            f"Among all groups, **{best}** performs the best with an average score of "
            f"**{group_means.max():.2f}**, while **{worst}** performs the weakest "
            f"with **{group_means.min():.2f}**."
        )

    # Outliers
    q1 = df[score_col].quantile(0.25)
    q3 = df[score_col].quantile(0.75)
    iqr = q3 - q1

    outliers = df[
        (df[score_col] < q1 - 1.5 * iqr) |
        (df[score_col] > q3 + 1.5 * iqr)
    ]

    if len(outliers) > 0:
        explanation.append(
            f"There are **{len(outliers)} unusual records** that may indicate "
            "exceptional performance or potential issues."
        )
    else:
        explanation.append(
            "No significant outliers were detected, suggesting consistent results."
        )

    return " ".join(explanation)

# ==================================================
# BASIC CHAT ANALYSIS (OFFLINE)
# ==================================================
def universal_analyze(df, prompt):
    prompt = prompt.lower()
    score_col = detect_score_column(df)
    group_col = detect_group_column(df)

    if 'report' in prompt or 'summary' in prompt:
        return f"""
📊 **DATA SUMMARY**

• Records: {len(df)}
• Columns: {len(df.columns)}
• Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}
• Text Columns: {len(df.select_dtypes(include='object').columns)}
"""

    if 'best' in prompt and group_col:
        best = df.groupby(group_col)[score_col].mean().idxmax()
        return f"📈 Best performing group: **{best}**"

    if 'worst' in prompt and group_col:
        worst = df.groupby(group_col)[score_col].mean().idxmin()
        return f"📉 Worst performing group: **{worst}**"

    return "💡 Try asking: full report, best group, worst group"

# ==================================================
# UI
# ==================================================
st.sidebar.title("📁 Upload CSV")
file = st.sidebar.file_uploader("Upload any CSV file", type="csv")

st.title("🎛️ Universal Forms AI Analyzer")
st.markdown("Employee • Customer • Student • Sales • **ANY CSV**")

if file:
    df = pd.read_csv(file)

    # Preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", len(df))
    col2.metric("Columns", len(df.columns))

    score_col = detect_score_column(df)
    group_col = detect_group_column(df)

    if score_col:
        col3.metric("Avg Score", f"{df[score_col].mean():.2f}")

    # AI Charts
    st.subheader("📈 AI-Generated Charts")
    charts = ai_chart_engine(df, score_col, group_col)
    for chart in charts:
        st.altair_chart(chart, use_container_width=True)

    heatmap = ai_correlation_heatmap(df)
    if heatmap:
        st.altair_chart(heatmap, use_container_width=True)

    # AI Explanation
    st.subheader("🤖 AI Explanation")
    st.markdown(ai_explain_insights(df, score_col, group_col))

    # Chat
    st.subheader("💬 Ask Your Data")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask: full report, best group, worst group"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            response = universal_analyze(df, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👈 Upload a CSV file to start AI analysis")

st.markdown("---")
st.caption("🚀 Offline AI • Natural Language Insights • No APIs Required")
