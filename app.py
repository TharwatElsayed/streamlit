import streamlit as st
from transformers import pipeline

# Fix for Keras 3 incompatibility (must install tf-keras)
# pip install tf-keras

# Mapping model output labels to readable class names
label_map = {
    "LABEL_0": "Neutral",
    "LABEL_1": "Offensive",
    "LABEL_2": "Hate"
}

@st.cache_resource
def load_pipeline():
    pipe = pipeline(
        "text-classification",
        model="ctoraman/hate-speech-bert"
    )
    return pipe

pipe = load_pipeline()

st.title("Hate Speech Detection")

text_input = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter text first.")
    else:
        with st.spinner("Analyzing..."):
            result = pipe(text_input)

        raw_label = result[0]["label"]
        readable_label = label_map.get(raw_label, raw_label)

        st.subheader("Prediction:")
        st.write(f"**Class:** {readable_label}")
