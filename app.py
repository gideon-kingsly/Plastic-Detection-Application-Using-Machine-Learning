import os
import shutil
import time
from collections import Counter
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="Plastic in River")
st.write("# Detect whether there is plastic in river or not")

@st.cache_resource
def load_model():
    """
    Load YOLOv8 pretrained model (COCO dataset)
    """
    model_path = "yolov8n.pt"
    return YOLO(model_path)

def get_predictions(model, image) -> Image:
    """
    Run YOLO prediction on uploaded image
    """
    results = model.predict(image, conf=0.25)
    res_img = results[0].plot(line_width=1)  # numpy array BGR
    res_img = res_img[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(res_img), results[0]

def get_pred_labels(results) -> dict:
    """
    Extract predicted labels and counts from YOLO results
    """
    names = results.names  # class id → label name
    detected = [names[int(box.cls)] for box in results.boxes]
    return dict(Counter(detected))

with st.sidebar:
    st.title("Plastic in River")
    st.sidebar.write(
        "Upload an image to predict whether there are any plastics "
        "(bottles, cups, bags, etc.) in the image."
    )
    with st.form("my_form"):
        if 'model' not in st.session_state:
            with st.spinner("Loading the model, please wait..."):
                model = load_model()
                st.session_state['model'] = model
                success_msg = st.success("Successfully loaded the model!")
            time.sleep(2)
            success_msg.empty()

        uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Predict")

if not submitted or not uploaded_image:
    st.stop()
else:
    try:
        # Convert uploaded file to PIL
        image = Image.open(uploaded_image)
        predicted_image, results = get_predictions(st.session_state['model'], image)
        predicted_labels = get_pred_labels(results)

        if predicted_labels:
            tab = "&ensp;"
            predictions = f",{tab}".join(f"{k}: {v}" for k, v in predicted_labels.items())
            st.info(f"PREDICTIONS {tab} → {tab} {predictions}")
        else:
            st.warning("No objects detected in this image.")

        st.image(predicted_image, caption="Detection Results", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
