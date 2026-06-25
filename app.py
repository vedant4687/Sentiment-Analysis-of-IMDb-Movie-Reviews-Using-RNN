import streamlit as st
import pandas as pd
from predict import predict_sentiment

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Mood Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD CSS ----------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------- SESSION STATE ----------------

if "sample" not in st.session_state:
    st.session_state.sample = ""

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------

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

st.sidebar.link_button(
    "💻 GitHub Repository",
    "https://github.com/vedant4687/Sentiment-Analysis-of-IMDb-Movie-Reviews-Using-RNN"
)

st.markdown("""
<div style='
text-align:center;
padding:25px;
border-radius:20px;
background:linear-gradient(135deg,#E50914,#7a0d13);
margin-bottom:20px;
'>
<h1 style='color:white;'>🎬 Mood Analyzer</h1>
<p style='color:white;font-size:18px;'>
Analyze IMDb Movie Reviews using Deep Learning & NLP
</p>
</div>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------


# ---------------- METRICS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "50K")

with col2:
    st.metric("Accuracy", "89%")

with col3:
    st.metric("Features", "5000")

st.markdown("---")

# ---------------- SAMPLE REVIEWS ----------------

st.subheader("✨ Sample Reviews")

col1, col2 = st.columns(2)

with col1:
    if st.button("😊 Positive Sample"):
        st.session_state.sample = (
            "This movie was absolutely fantastic and I loved every minute of it."
        )

with col2:
    if st.button("☹️ Negative Sample"):
        st.session_state.sample = (
            "Terrible acting and horrible storyline. Worst movie ever."
        )

# ---------------- INPUT ----------------

review = st.text_area(
    "✍ Enter your movie review",
    value=st.session_state.sample,
    height=200
)

# ---------------- MODEL DETAILS ----------------

with st.expander("📊 Model Details"):
    st.write("""
    Architecture : RNN

    Framework : PyTorch

    Feature Extraction : TF-IDF

    Vocabulary Size : 5000

    Dataset : IMDb 50K Reviews
    """)

# ---------------- PREDICTION ----------------

if st.button("🔍 Analyze Mood"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        sentiment, confidence = predict_sentiment(review)

        # Save History
        st.session_state.history.append(
            {
                "Review": review[:50],
                "Sentiment": sentiment,
                "Confidence (%)": round(confidence * 100, 2)
            }
        )

        # Result Card
        if sentiment == "Positive":

            st.markdown(
                f"""
                <div class='positive'>
                <h2>😊 Positive Review</h2>
                <h3>Confidence : {confidence*100:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class='negative'>
                <h2>☹️ Negative Review</h2>
                <h3>Confidence : {confidence*100:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Confidence Bar

        st.subheader("📊 Confidence Score")

        st.progress(float(confidence))

        st.success(
            f"Model Confidence: {confidence*100:.2f}%"
        )

# ---------------- HISTORY ----------------

if len(st.session_state.history) > 0:

    st.markdown("---")

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history[::-1]
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <h4>🎬 Mood Analyzer</h4>
        <p>Built with ❤️ using PyTorch, Streamlit and NLP</p>
        <p>Developed by <b>Vedant Deshmukh</b></p>
        <p>
            <a href="https://github.com/vedant4687" target="_blank">
            🔗 GitHub Profile
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)