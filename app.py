import streamlit as st
import cv2
import numpy as np
import joblib
import pywt
from skimage.feature import hog
from PIL import Image
import os

# Load models
logistic_model = joblib.load("celebrity_face_lr_model.pkl")
svm_model = joblib.load("celebrity_face_svm_model.pkl")

# Class dictionary
class_dict = {
    'Angelina Jolie': 0,
    'Brad Pitt': 1,
    'Jennifer Lawrence': 2,
    'Johnny Depp': 3,
    'Megan Fox': 4,
    'Robert Downey Jr': 5,
    'Sandra Bullock': 6,
    'Tom Cruise': 7,
    'Tom Hanks': 8,
    'Will Smith': 9
}
inv_class_dict = {v: k for k, v in class_dict.items()}

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Wavelet function
def w2d(img, mode='haar', level=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(gray) / 255.0
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)
    return imArray_H

# HOG features
def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    return features

# Combined feature extraction
def extract_features(img):
    resized_img = cv2.resize(img, (64, 64))
    wavelet_img = w2d(img, 'db1', 5)
    resized_wavelet = cv2.resize(wavelet_img, (64, 64))
    wavelet_features = resized_wavelet.flatten()
    hog_features = extract_hog_features(resized_img)
    combined_features = np.hstack((wavelet_features, hog_features))
    return combined_features

# Cropping the image else returning none
def get_cropped_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Tweaked scaleFactor
    for (x, y, w, h) in faces:
        cropped_img = img[y:y+h, x:x+w]
        return cv2.resize(cropped_img, (250, 250))
    return None


# Streamlit app
st.set_page_config(page_title="Celebrity Face Classifier", layout="wide")
st.title("🎬 Celebrity Face Classifier")

# Display celebrity grid
st.subheader("🌟 Our Celebrities")
sample_faces_dir = "sample_faces"
cols = st.columns(10)
for i, (name, _) in enumerate(class_dict.items()):
    col = cols[i % 10]
    img_path = os.path.join(sample_faces_dir, f"{name}.jpg")
    try:
        image = Image.open(img_path)
        col.image(image, caption=name, use_container_width=True)
    except:
        col.warning(f"Missing: {name}")

# Sidebar
st.sidebar.header("⚙️ Model Selection")
model_option = st.sidebar.radio("Choose a Model", ["Logistic Regression", "SVM"])
uploaded_file = st.sidebar.file_uploader("📤 Upload a Face Image", type=['jpg', 'jpeg', 'png'])
predict_btn = st.sidebar.button("🔮 Predict")

# Main content
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption='Uploaded Image')

    if predict_btn:
        cropped_img = get_cropped_face(img)
        if cropped_img is not None:
            try:
                features = extract_features(cropped_img).reshape(1, -1)
                model = logistic_model if model_option == "Logistic Regression" else svm_model
                prediction = model.predict(features)[0]
                st.success(f"🎉 Prediction: **{inv_class_dict[prediction]}**")
            except Exception as e:
                st.error(f"Couldn't process the image: {e}")
        else:
            st.warning("⚠️ Couldn't find a clear face with 2 eyes. Please try another image.")
