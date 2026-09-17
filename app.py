import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(
    page_title="Waste sorter",
    page_icon=":material/recycling:",
    layout="centered",
)


@st.cache_resource
def load_model_bundle():
    model = keras.models.load_model(MODEL_DIR / "waste_classifier.keras")
    class_names = json.loads((MODEL_DIR / "class_names.json").read_text())
    config = json.loads((MODEL_DIR / "model_config.json").read_text())
    return model, class_names, config["img_size"]


def predict(image: Image.Image, model, class_names, img_size):
    image = image.convert("RGB").resize((img_size, img_size))
    array = np.expand_dims(np.array(image), axis=0).astype("float32")
    array = preprocess_input(array)
    probs = model.predict(array, verbose=0)[0]
    return probs


st.title("Waste sorter")
st.caption("Upload a photo of an item and get its predicted waste category.")

model, class_names, img_size = load_model_bundle()

uploaded_file = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_image, col_result = st.columns(2, gap="large")

    with col_image:
        st.image(image, width="stretch")

    with col_result:
        with st.spinner("Classifying..."):
            probs = predict(image, model, class_names, img_size)

        top_idx = int(np.argmax(probs))
        top_label = class_names[top_idx]
        top_conf = float(probs[top_idx])

        st.metric("Predicted category", top_label.replace("-", " ").title())
        st.progress(top_conf, text=f"Confidence: {top_conf:.1%}")

        st.write("")
        st.caption("Top candidates")
        order = np.argsort(probs)[::-1][:5]
        chart_df = pd.DataFrame({
            "category": [class_names[i].replace("-", " ").title() for i in order],
            "probability": [float(probs[i]) for i in order],
        }).set_index("category")
        st.bar_chart(chart_df, horizontal=True)
else:
    st.info(
        "No image uploaded yet — drop in a photo of cardboard, glass, metal, "
        "paper, plastic, biological waste, a battery, clothes, shoes, or general trash.",
        icon=":material/upload_file:",
    )
