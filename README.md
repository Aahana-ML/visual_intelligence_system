# 🔭 Visual Intelligence System

A combined computer vision system that performs **scene classification** and **object detection** on a single image.

The system combines a fine-tuned **EfficientNetB0** model for scene classification with a **YOLO** object detection model, allowing it to understand both the overall environment and the individual objects present in an image.

## 🚀 Live Demo

[Try the deployed Visual Intelligence System](YOUR_STREAMLIT_URL_HERE)

---

## 📌 Project Overview

Traditional computer vision systems often focus on a single task, such as identifying objects or classifying an entire image.

This project combines both approaches:

- 🌍 **Scene Classification** — identifies the overall scene/environment.
- 🎯 **Object Detection** — identifies multiple objects and their locations.
- 🔗 **Combined Analysis** — presents both types of information together in one application.

For example, an image of a city street can be classified as **street** while simultaneously detecting objects such as cars, buses, bicycles, people, and motorbikes.

---

## 🧠 Models Used

### 🌍 Scene Classification — EfficientNetB0

A fine-tuned **EfficientNetB0** model is used to classify images into six scene categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The model was trained using transfer learning followed by fine-tuning.

### 🎯 Object Detection — YOLO

A YOLO-based object detection model is used to detect multiple objects within an image.

The detector identifies objects from the VOC-style object classes used during training and provides:

- Object class
- Confidence score
- Bounding box

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