import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# Page config
st.set_page_config(page_title="📊 Universal Forms AI Analyzer", layout="wide")

def universal_analyze(df, prompt):
    """Smart analysis for ANY CSV form data"""
    prompt_lower = prompt.lower()
    
    # Auto-detect columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Smart column detection
    score_col = None
    group_col = None
    feedback_col = None
    
    for col in numeric_cols:
        if any(x in col.lower() for x in ['score', 'rate', 'satisf', 'rating', 'mark', 'grade']):
            score_col = col
            break
    else:
        score_col = numeric_cols[0] if numeric_cols else None
    
    for col in text_cols:
        if any(x in col.lower() for x in ['group', 'dept', 'team', 'class', 'division', 'department']):
            group_col = col
            break
    
    for col in text_cols:
        if any(x in col.lower() for x in ['feedback', 'comment', 'remark', 'note', 'review']):
            feedback_col = col
            break
    
    # RESPONSE LOGIC
    if 'depart' in prompt_lower or 'group' in prompt_lower or 'team' in prompt_lower:
        if group_col and group_col in df.columns and score_col:
            group_stats = df.groupby(group_col)[score_col].agg(['mean', 'count']).round(1)
            group_stats.columns = ['Avg Score', 'Count']
            return f"""
**🏢 GROUPS/DEPARTMENTS ANALYSIS:**
{group_stats.to_string()}
• **Total Groups**: {df[group_col].nunique()}
• **Overall Avg**: {df[score_col].mean():.1f}"""
        else:
            return f"**📋 COLUMNS**: {', '.join(df.columns.tolist())}"
    
    elif 'low' in prompt_lower or 'worst' in prompt_lower:
        if score_col and group_col and group_col in df.columns:
            lowest_group = df.groupby(group_col)[score_col].mean().idxmin()
            lowest_score = df.groupby(group_col)[score_col].mean().min()
            return f"""
**📉 LOWEST PERFORMING:**
• **{lowest_group}**: {lowest_score:.1f} ({df[df[group_col]==lowest_group][score_col].count()} responses)
• **Overall Avg**: {df[score_col].mean():.1f}"""
        else:
            return f"**📉 LOWEST SCORE**: {df[numeric_cols[0]].min():.1f}"
    
    elif 'high' in prompt_lower or 'best' in prompt_lower:
        if score_col and group_col and group_col in df.columns:
            highest_group = df.groupby(group_col)[score_col].mean().idxmax()
            highest_score = df.groupby(group_col)[score_col].mean().max()
            return f"""
**📈 HIGHEST PERFORMING:**
• **{highest_group}**: {highest_score:.1f} ({df[df[group_col]==highest_group][score_col].count()} responses)
• **Overall Avg**: {df[score_col].mean():.1f}"""
        else:
            return f"**📈 HIGHEST SCORE**: {df[numeric_cols[0]].max():.1f}"
    
    elif 'report' in prompt_lower or 'analyze' in prompt_lower or 'summary' in prompt_lower:
        summary = f"""
**📊 COMPLETE DATA REPORT** ({len(df)} records)

**📋 COLUMNS DETECTED** ({len(df.columns)} total):
"""
        if numeric_cols:
            summary += f"• **Numeric** ({len(numeric_cols)}): {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols)>3 else ''}\n"
        if text_cols:
            summary += f"• **Text** ({len(text_cols)}): {', '.join(text_cols[:3])}{'...' if len(text_cols)>3 else ''}\n"
        
        if numeric_cols:
            summary += f"\n**🔢 KEY METRICS** (Top 3 numeric columns):\n"
            for col in numeric_cols[:3]:
                summary += f"• **{col}**: Avg={df[col].mean():.1f}, Min={df[col].min():.0f}, Max={df[col].max():.0f}\n"
        
        if group_col:
            summary += f"\n**🏢 GROUPS FOUND**: {df[group_col].nunique()} ({', '.join(df[group_col].unique()[:3])}{'...' if len(df[group_col].unique())>3 else ''})"
        
        return summary
    
    elif 'trend' in prompt_lower or 'pattern' in prompt_lower or 'correlat' in prompt_lower:
        if len(numeric_cols) >= 2:
            correlations = df[numeric_cols[:3]].corr()
            strong_corr = correlations.abs().stack().drop_duplicates().nlargest(3).round(2)
            return f"""
**📈 TRENDS & CORRELATIONS:**
• **Strongest**: {strong_corr.index[0][0]} ↔ {strong_corr.index[0][1]} (r={strong_corr.iloc[0]:.2f})
• **Records analyzed**: {len(df)}
• **Columns**: {len(numeric_cols)} numeric"""
        else:
            return f"**📈 TRENDS**: {len(df)} records, {len(numeric_cols)} numeric columns"
    
    else:
        return f"""
**🔍 INSTANT INSIGHTS** ({len(df)} records)

**📊 DATA STRUCTURE:**
• **Total Columns**: {len(df.columns)}
• **Numeric Fields**: {len(numeric_cols)} 
• **Text Fields**: {len(text_cols)}

**💡 ASK ABOUT:**
• "departments" "groups" "teams"
• "lowest scores" "worst performing" 
• "highest scores" "best performing"
• "full report" "analyze all"
• "trends" "correlations"
"""

# Sidebar
st.sidebar.title("📁 Upload ANY Form CSV")
uploaded_file = st.sidebar.file_uploader("Employee, Customer, Student, Sales...", type="csv")

# Main title
st.title("🎛️ Universal Forms Analyzer")
st.markdown("**Employee feedback • Customer surveys • Student forms • Sales data • ANY CSV!**")

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # ✅ FIXED: was 'uploadlined_file'
    
    # Data preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📈 Auto-Detected Stats")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("Records", len(df))
        if len(numeric_cols) > 0:
            st.metric("Avg Score", f"{df[numeric_cols[0]].mean():.1f}")
        st.metric("Columns", len(df.columns))
    
    # Chat interface
    st.subheader("💬 Ask About Your Data")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("departments? lowest scores? full report? trends?..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Smart Analysis..."):
                response = universal_analyze(df, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
else:
    st.info("👈 **UPLOAD ANY CSV → INSTANT SMART ANALYSIS!**")
    st.markdown("""
    **✅ Works Instantly With:**
    • Employee satisfaction surveys
    • Customer feedback forms  
    • Student evaluation forms
    • Sales performance reports
    • **ANY CSV with numbers/text**
    
    **💬 Smart Questions:**
    • "show departments" 
    • "lowest performing group"
    • "highest scores"
    • "complete report"
    • "find trends"
    """)

st.markdown("---")
st.markdown("*🚀 100% FREE • No APIs • Works with ALL CSV forms*")
