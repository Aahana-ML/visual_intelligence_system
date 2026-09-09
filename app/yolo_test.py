from ultralytics import YOLO
import streamlit as st

st.title("YOLO Test")

model = YOLO("models/yolo_detector.pt")

st.write("YOLO model loaded successfully.")

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    from PIL import Image

    image = Image.open(uploaded).convert("RGB")

    st.image(image)

    if st.button("Test YOLO"):

        st.write("Starting YOLO inference...")

        results = model(
            image,
            conf=0.4,
            device="cpu"
        )

        st.write("YOLO inference finished!")

        st.write(results[0].boxes)
