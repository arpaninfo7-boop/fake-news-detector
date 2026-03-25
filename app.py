import streamlit as st
from model import predict_news, predict_confidence

st.set_page_config(page_title="AI Fake News Detector", page_icon="🧠", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 AI Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect whether news is REAL or FAKE using AI</p>", unsafe_allow_html=True)

st.write("---")

# Input
user_input = st.text_area("✍️ Paste News Text Here:", height=200)

# Button
if st.button("🔍 Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        result = predict_news(user_input)
        confidence = predict_confidence(user_input)

        st.write("### 🧾 Result:")

        if result == "FAKE":
            st.error("🚨 This News is FAKE")
            reason = "This content contains sensational or misleading phrases often found in fake news."
        else:
            st.success("✅ This News is REAL")
            reason = "This content appears factual and similar to verified news patterns."

        # Confidence bar
        st.progress(int(confidence * 100))

        st.info(f"Confidence Score: {confidence*100:.2f}%")

        # Explanation
        st.write("### 🤖 AI Explanation:")
        st.write(reason)

st.write("---")
st.caption("Built using Machine Learning & Streamlit")
