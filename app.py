import streamlit as st
from transformers import pipeline

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_pipeline():
    pipe = pipeline("text-classification", model="ctoraman/hate-speech-bert")
    return pipe

pipe = load_pipeline()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Hate Speech Detection", layout="centered")

st.title("🛡️ Hate Speech Detection App")
st.write("This app uses the **ctoraman/hate-speech-bert** model to classify text.")

# Input text
text_input = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter text first.")
    else:
        with st.spinner("Analyzing..."):
            result = pipe(text_input)

        label = result[0]["label"]
        score = float(result[0]["score"])

        st.subheader("🔎 Prediction Result")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {score:.4f}")

        # Color-coded UI
        if "hate" in label.lower():
            st.error("⚠️ Hate Speech Detected")
        else:
            st.success("✔️ Not Hate Speech")
