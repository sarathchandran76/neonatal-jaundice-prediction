import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from huggingface_hub import hf_hub_download
import os

# Download model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Sarath892/neonatal-jaundice-model",
    filename="modelupdated.keras"
)

model = load_model(model_path)

IMG_SIZE = (224, 224)

# (Keep all your previous functions for feature extraction and LBP)
def extract_features(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    avg_R = np.mean(img_rgb[:, :, 0])
    avg_G = np.mean(img_rgb[:, :, 1])
    avg_B = np.mean(img_rgb[:, :, 2])

    avg_H = np.mean(img_hsv[:, :, 0])
    avg_S = np.mean(img_hsv[:, :, 1])
    avg_V = np.mean(img_hsv[:, :, 2])

    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    rg_ratio = avg_R / (avg_G + 1e-5)
    rb_ratio = avg_R / (avg_B + 1e-5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 256), density=True)

    features = [avg_R, avg_G, avg_B, avg_H, avg_S, avg_V, brightness, rg_ratio, rb_ratio]
    features.extend(lbp_hist.tolist())

    features = np.array(features)
    features = (features - features.mean()) / (features.std() + 1e-8)

    return features

def local_binary_pattern(gray):
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            center = gray[i, j]
            binary_str = ''
            binary_str += '1' if gray[i - 1, j - 1] >= center else '0'
            binary_str += '1' if gray[i - 1, j] >= center else '0'
            binary_str += '1' if gray[i - 1, j + 1] >= center else '0'
            binary_str += '1' if gray[i, j + 1] >= center else '0'
            binary_str += '1' if gray[i + 1, j + 1] >= center else '0'
            binary_str += '1' if gray[i + 1, j] >= center else '0'
            binary_str += '1' if gray[i + 1, j - 1] >= center else '0'
            binary_str += '1' if gray[i, j - 1] >= center else '0'
            lbp[i, j] = int(binary_str, 2)
    return lbp

# Streamlit interface
st.title("Neonatal Jaundice Prediction")
st.write("Upload a neonatal skin image to predict bilirubin level.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Bilirubin Level"):
        resized_image = cv2.resize(image, IMG_SIZE)
        resnet_input = preprocess_input(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        resnet_input = np.expand_dims(resnet_input, axis=0)

        features = extract_features(image)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict([resnet_input, features])
        st.success(f"Predicted Serum Bilirubin Level: {prediction[0][0]:.2f} mg/dL")
