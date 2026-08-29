# 🔭 Visual Intelligence System

A combined computer vision system that performs **scene classification** and **object detection** on a single image.

The system combines a fine-tuned **EfficientNetB0** model for scene classification with a **YOLO** object detection model, allowing it to understand both the overall environment and the individual objects present in an image.

## 🚀 Live Demo

[Try the deployed Visual Intelligence System](https://visualintelligencesystem-nwvcrzwb9cvbeeflbx2kzr.streamlit.app/)

---

## 📌 Project Overview

Traditional computer vision systems often focus on a single task, such as identifying objects or classifying an entire image.

This project combines both approaches into a single visual intelligence system:

- 🌍 **Scene Classification** — identifies the overall scene or environment.
- 🎯 **Object Detection** — identifies multiple objects and their locations.
- 🔗 **Combined Analysis** — presents both types of information together in one application.

For example, an image of a city street can be classified as **street** while simultaneously detecting objects such as cars, buses, bicycles, people, and motorbikes.

---

## 🧠 Models Used

### 🌍 Scene Classification — Fine-Tuned EfficientNetB0

A fine-tuned **EfficientNetB0** model is used to classify images into six scene categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The model was developed using **transfer learning** followed by **fine-tuning**.

### 🎯 Object Detection — YOLO

A YOLO-based object detection model is used to detect multiple objects within an image.

The detector was trained using **20 Pascal VOC object classes** and provides:

- Object class
- Confidence score
- Bounding box

The supported object classes are:

- Aeroplane
- Bicycle
- Bird
- Boat
- Bottle
- Bus
- Car
- Cat
- Chair
- Cow
- Dining Table
- Dog
- Horse
- Motorbike
- Person
- Potted Plant
- Sheep
- Sofa
- Train
- TV Monitor

---

## 🏗️ System Architecture

```text
                    Input Image
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       EfficientNetB0             YOLO
       Scene Classifier        Object Detector
              │                     │
              ▼                     ▼
       Predicted Scene       Detected Objects
       + Confidence          + Confidence
                                    │
              └──────────┬──────────┘
                         ▼
              Combined Visual Result
                         │
                         ▼
                  Streamlit App
```

The input image is processed by both models independently. EfficientNetB0 identifies the overall scene, while YOLO detects individual objects. Their outputs are then presented together through the Streamlit application.

---

## 📊 Model Performance

### 🎯 YOLO Object Detection

Final evaluation metrics:

| Metric | Score |
|--------|-------|
| Precision | 74.13% |
| Recall | 64.69% |
| mAP@50 | 71.49% |
| mAP@50–95 | 53.64% |

### 🌍 Scene Classification

The fine-tuned EfficientNetB0 model was evaluated on a test set containing:

**3,000 images across 6 scene classes.**

Final test accuracy:

**92.70%**

The model performed particularly well on scenes such as:

- 🌲 Forest
- 🌊 Sea
- 🏙️ Street
- 🏢 Buildings

Some confusion was observed between visually similar categories, particularly **glacier and mountain**.

---

## 🖼️ Example

The system can analyze an image and produce results such as:

```text
Scene: Street
Scene confidence: 100.00%

Objects detected:
- bus: 0.95
- person: 0.97
- person: 0.95
- car: 0.88
- car: 0.86
- bicycle: 0.65
- motorbike: ...
```

The application also displays the detected objects with bounding boxes directly on the image.

---

## 🔍 Features

- 📤 Image upload through a web interface
- 🎚️ Adjustable YOLO confidence threshold
- 🎯 Multiple object detection
- 📦 Bounding-box visualization
- 🌍 Scene classification with confidence score
- 🔗 Combined scene and object analysis
- 🖼️ Annotated image visualization
- 🌐 Interactive Streamlit interface
- ☁️ Cloud deployment

---

## 💻 Technologies

- **Python**
- **TensorFlow / Keras**
- **EfficientNetB0**
- **YOLO**
- **Ultralytics**
- **OpenCV**
- **NumPy**
- **Pillow**
- **Matplotlib**
- **Streamlit**
- **Git**
- **GitHub**

---

## 📁 Project Structure

```text
visual_intelligence_system/
│
├── app/
│   └── app.py
│
├── images/
│   └── test images
│
├── models/
│   ├── scene_classifier_efficientnetb0_finetuned.keras
│   └── yolo_detector.pt
│
├── notebooks/
│   ├── 01_scene_data_exploration.ipynb
│   └── 02_model_combination.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Aahana-ML/visual_intelligence_system.git
cd visual_intelligence_system
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## ⚠️ Limitations

### 🌍 Scene Classifier

The scene classifier currently supports only six scene categories:

- 🏢 Buildings
- 🌲 Forest
- 🧊 Glacier
- ⛰️ Mountain
- 🌊 Sea
- 🏙️ Street

There is **no background or unknown class**.

Therefore, if an input image does not belong to any of these six categories, the classifier will still assign it to the class with the highest predicted probability.

For example, an image of an office, kitchen, close-up portrait, or another unseen environment may still be classified as one of the six available classes.

This means predictions on completely unseen environments may not always be reliable.

### 🎯 Object Detector

Object detection performance can vary depending on:

- Image quality
- Object size
- Lighting conditions
- Occlusion
- Unusual viewpoints
- Objects that are difficult to distinguish visually

The YOLO detector may also produce false positives or miss objects, particularly when objects are very small or heavily occluded.

### 🧩 Model Architecture

The scene classifier and object detector are independent models.

They are combined at the **application level** rather than being trained together as a single multimodal architecture.

---

## 🔮 Future Improvements

Possible improvements include:

- Adding an **unknown/background** option for scene classification
- Expanding the number of scene categories
- Improving detection of small objects
- Training a unified multimodal architecture
- Adding video and webcam inference
- Adding object counting
- Generating automatic scene descriptions
- Improving confidence calibration
- Optimizing inference speed
- Adding more diverse training data

---

## 🤝 Acknowledgements

This project was developed as an AI/ML learning project with guidance and learning support from **ChatGPT**.

---

## 👩‍💻 Author

**Aahana**

GitHub: [Aahana-ML](https://github.com/Aahana-ML)