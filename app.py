import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="📊 Universal Forms AI Analyzer",
    layout="wide"
)

# =========================
# SMART COLUMN DETECTION
# =========================
def detect_score_column(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None

    keywords = ['score', 'rate', 'rating', 'satisf', 'mark', 'grade']
    for col in numeric_cols:
        if any(k in col.lower() for k in keywords):
            return col

    # fallback → highest variance column
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

# =========================
# FEEDBACK ANALYSIS
# =========================
def analyze_feedback(df, feedback_col):
    positive = ['good', 'great', 'excellent', 'happy', 'satisfied', 'love']
    negative = ['bad', 'poor', 'worst', 'slow', 'unsatisfied', 'hate']

    text = df[feedback_col].dropna().str.lower()
    pos = sum(text.str.contains('|'.join(positive)))
    neg = sum(text.str.contains('|'.join(negative)))

    return pos, neg

# =========================
# CHARTS
# =========================
def plot_group_scores(df, group_col, score_col):
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(group_col, sort='-y'),
        y=alt.Y(score_col, aggregate='mean'),
        tooltip=[group_col, alt.Tooltip(score_col, aggregate='mean')]
    ).properties(title="Average Score by Group")
    return chart

# =========================
# CORE ANALYSIS ENGINE
# =========================
def universal_analyze(df, prompt):
    prompt = prompt.lower()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include='object').columns.tolist()

    score_col = detect_score_column(df)
    group_col = detect_group_column(df)
    feedback_col = detect_feedback_column(df)

    # -------------------------
    if 'depart' in prompt or 'group' in prompt or 'team' in prompt:
        if group_col and score_col:
            stats = df.groupby(group_col)[score_col].agg(['mean', 'count']).round(2)
            return f"""
**🏢 GROUP ANALYSIS**
{stats.to_string()}

• Total Groups: {df[group_col].nunique()}
• Overall Avg: {df[score_col].mean():.2f}
"""
        return f"Detected columns: {', '.join(df.columns)}"

    # -------------------------
    if 'low' in prompt or 'worst' in prompt:
        if group_col and score_col:
            worst = df.groupby(group_col)[score_col].mean().idxmin()
            return f"📉 Worst Performing Group: **{worst}** ({df[df[group_col]==worst][score_col].mean():.2f})"
        return f"📉 Lowest Value: {df[numeric_cols[0]].min()}"

    # -------------------------
    if 'high' in prompt or 'best' in prompt:
        if group_col and score_col:
            best = df.groupby(group_col)[score_col].mean().idxmax()
            return f"📈 Best Performing Group: **{best}** ({df[df[group_col]==best][score_col].mean():.2f})"
        return f"📈 Highest Value: {df[numeric_cols[0]].max()}"

    # -------------------------
    if 'trend' in prompt or 'correlat' in prompt:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            corr = corr.where(~np.eye(corr.shape[0], dtype=bool))
            strongest = corr.abs().unstack().dropna().sort_values(ascending=False).head(3)
            return f"""
**📈 STRONG CORRELATIONS**
{strongest.to_string()}
"""
        return "Not enough numeric columns for correlation."

    # -------------------------
    if 'report' in prompt or 'summary' in prompt or 'analyze' in prompt:
        report = f"""
📊 **FULL DATA REPORT**

• Records: {len(df)}
• Columns: {len(df.columns)}
• Numeric: {len(numeric_cols)}
• Text: {len(text_cols)}

"""
        if score_col:
            report += f"""
🔢 **Score Column: {score_col}**
• Avg: {df[score_col].mean():.2f}
• Min: {df[score_col].min()}
• Max: {df[score_col].max()}
"""

        if group_col:
            report += f"\n🏢 Groups Detected: {df[group_col].nunique()}"

        if feedback_col:
            pos, neg = analyze_feedback(df, feedback_col)
            report += f"\n📝 Feedback → Positive: {pos}, Negative: {neg}"

        return report

    # -------------------------
    return f"""
🔍 **INSTANT INSIGHTS**
• Records: {len(df)}
• Numeric Columns: {len(numeric_cols)}
• Text Columns: {len(text_cols)}

💡 Try asking:
- show departments
- worst performing group
- best team
- full report
- trends & correlations
"""

# =========================
# UI
# =========================
st.sidebar.title("📁 Upload CSV")
file = st.sidebar.file_uploader("Upload any CSV file", type="csv")

st.title("🎛️ Universal Forms Analyzer")
st.markdown("Employee • Customer • Student • Sales • ANY CSV")

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
    if score_col:
        col3.metric("Avg Score", f"{df[score_col].mean():.2f}")

    # Chart
    group_col = detect_group_column(df)
    if group_col and score_col:
        st.altair_chart(plot_group_scores(df, group_col, score_col), use_container_width=True)

    # Chat
    st.subheader("💬 Ask Your Data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something like: full report, worst group, trends"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = universal_analyze(df, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Download report
    report = universal_analyze(df, "full report")
    st.download_button("📥 Download Report", report, file_name="analysis_report.txt")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

else:
    st.info("👈 Upload any CSV file to start analysis")

st.markdown("---")
st.caption("🚀 Universal Forms AI Analyzer • No APIs • Fully Offline")
