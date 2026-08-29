import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Visual Intelligence System",
    page_icon="🔭",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🔭 Visual Intelligence System")
st.caption("Scene Classification • Object Detection • Visual Understanding")


st.write(
    "Upload an image to classify its scene "
    "and detect objects using deep learning."
)

st.sidebar.header("Detection Settings")

confidence_threshold = st.sidebar.slider(
    "YOLO Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.subheader("About")

st.sidebar.markdown(
    """
    This system combines two deep learning models:

    🎯 **YOLO**  
    Object detection

    🌍 **EfficientNetB0**  
    Scene classification

    Together, they provide object-level and
    scene-level understanding.
    """
)


# ==========================================
# MODEL PATHS
# ==========================================

YOLO_PATH = "models/yolo_detector.pt"

SCENE_MODEL_PATH = (
    "models/scene_classifier_efficientnetb0_finetuned.keras"
)


# ==========================================
# SCENE CLASSES
# ==========================================

scene_classes = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    yolo_model = YOLO(YOLO_PATH)

    scene_model = load_model(
        SCENE_MODEL_PATH
    )

    return yolo_model, scene_model


yolo_model, scene_model = load_models()


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    analyze_button = st.button(
        "🔍 Analyze Image"
    )

    if analyze_button:
        # ==========================================
        # YOLO OBJECT DETECTION
        # ==========================================

        yolo_results = yolo_model(
            image,
            conf=confidence_threshold
        )

        yolo_result = yolo_results[0]

        # ==========================================
        # SCENE CLASSIFICATION
        # ==========================================

        img = image.resize((150, 150))

        img_array = img_to_array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # IMPORTANT:
        # No /255 here.
        # This matches our trained model pipeline.

        scene_predictions = scene_model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            scene_predictions[0]
        )

        predicted_scene = scene_classes[
            predicted_index
        ]

        scene_confidence = float(
            scene_predictions[0][predicted_index]
        )

        # ==========================================
        # SCENE RESULT
        # ==========================================

        st.subheader("🌍 Scene Classification")

        col1, col2 = st.columns(2)

        with col1:
            st.info(
                f"**Predicted Scene**\n\n"
                f"### {predicted_scene.title()}"
            )

        with col2:
            st.success(
                f"**Confidence**\n\n"
                f"### {scene_confidence:.2%}"
            )

        # ==========================================
        # OBJECT DETECTIONS
        # ==========================================

        st.subheader("🎯 Detected Objects")

        detected_objects = []

        for conf, cls in zip(
            yolo_result.boxes.conf.cpu().numpy(),
            yolo_result.boxes.cls.cpu().numpy()
        ):

            class_name = yolo_model.names[int(cls)]

            detected_objects.append(
                (class_name, float(conf))
            )

        if detected_objects:

            for class_name, confidence in detected_objects:

                st.write(
                    f"**{class_name}** — "
                    f"{confidence:.2%}"
                )

            st.write(
                f"**Total objects detected:** "
                f"{len(detected_objects)}"
            )

        else:

            st.write("No objects detected.")

        # ==========================================
        # YOLO VISUALIZATION
        # ==========================================

        st.subheader("🔍 Object Detection")

        annotated_image = yolo_result.plot()

        st.image(
            annotated_image[:, :, ::-1],
            caption="YOLO Object Detections",
            use_container_width=True
        )

        st.markdown("---")

        st.markdown(
    """
    <div style="
        text-align: center;
        font-size: 0.75rem;
        color: yellow;
    ">
        Visual Intelligence System ·
        Built with YOLO, EfficientNetB0 & Streamlit ·
        Developed with guidance from ChatGPT
    </div>
    """,
    unsafe_allow_html=True
)