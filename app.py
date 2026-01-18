import streamlit as st
import pandas as pd
from openai import OpenAI

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📊 Student Forms AI Analyzer",
    layout="wide"
)

# ---------------- OPENAI CLIENT ----------------
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("❌ OpenAI API key not found. Please add it in Streamlit Secrets.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Setup")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV forms",
    type="csv"
)

# ---------------- MAIN TITLE ----------------
st.title("🤖 Student Forms AI Analyzer")
st.markdown("**Ask any question about your student forms data!**")

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------- DATA PREVIEW --------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("📈 Quick Stats")
        st.metric("Total Forms", len(df))
        st.metric(
            "Avg Satisfaction",
            f"{df.get('satisfaction', pd.Series([0])).mean():.1f}/10"
        )
        st.metric(
            "Departments",
            df.get('department', pd.Series(['N/A'])).nunique()
        )

    # -------- CHAT SECTION --------
    st.subheader("💬 Ask Questions About Your Data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(
        "Ask about trends, satisfaction, departments, feedback..."
    ):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # -------- DATA FOR AI --------
        sample_data = df.sample(
            min(20, len(df))
        ).to_csv(index=False)

        stats_summary = {
            "rows": len(df),
            "avg_satisfaction": df.get(
                "satisfaction", pd.Series([0])
            ).mean(),
            "departments": df.get(
                "department", pd.Series(["N/A"])
            ).value_counts().to_dict()
        }

        ai_prompt = f"""
You are analyzing student or office feedback form data.

SAMPLE DATA:
{sample_data}

STATS SUMMARY:
{stats_summary}

QUESTION:
{prompt}

Give concise, data-driven insights in bullet points.
"""

        # -------- AI RESPONSE --------
        with st.chat_message("assistant"):
            with st.spinner("AI is analyzing your data..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": ai_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=400
                    )

                    answer = response.choices[0].message.content
                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")

else:
    st.info("👈 Upload a CSV file from the sidebar to start!")
    st.markdown("""
### Example questions you can ask:
- What are the main feedback themes?
- Which department has the lowest satisfaction?
- Any trends in complaints?
- Actionable recommendations to improve scores?
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "*Built with Streamlit + OpenAI | Secure • Cloud-Ready • College-Friendly*"
)
