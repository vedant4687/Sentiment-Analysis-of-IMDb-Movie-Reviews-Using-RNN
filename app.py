import plotly.graph_objects as go
import streamlit as st
from predict import predict_sentiment


st.set_page_config(
    page_title="Mood Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Sidebar
st.sidebar.title("📊 Model Information")

st.sidebar.markdown("""
---
### 🎬 Dataset
50K IMDb Reviews

### 🧠 Model
RNN (PyTorch)

### 🔢 Features
TF-IDF (5000)

### 📈 Accuracy
~89%

### ⚙ Framework
PyTorch + Streamlit
---
""")

# Load CSS
with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

st.markdown(
"""
<div class='title'>
🎬 Mood Analyzer
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
AI-powered IMDb Movie Review Sentiment Analysis
</div>
""",
unsafe_allow_html=True
)

st.subheader("✨ Sample Reviews")

col1, col2 = st.columns(2)

with col1:
    if st.button("😊 Positive Sample"):
        st.session_state.sample = (
            "This movie was absolutely fantastic and I loved every minute of it."
        )

with col2:
    if st.button("☹ Negative Sample"):
        st.session_state.sample = (
            "Terrible acting and horrible storyline. Worst movie ever."
        )

if "sample" not in st.session_state:
    st.session_state.sample = ""

review = st.text_area(
    "✍ Enter your movie review",
    height=200
)


    # Confidence Gauge
fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": "Confidence"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#E50914"},
            "steps": [
                {"range": [0, 50], "color": "#303030"},
                {"range": [50, 100], "color": "#505050"}
            ]
        }
    )
)

fig.update_layout(
    paper_bgcolor="#141414",
    font={"color": "white"}
)

st.plotly_chart(fig, use_container_width=True)

