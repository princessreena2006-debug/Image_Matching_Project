import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Face Matching", layout="centered")
st.title("Face Detection & Matching")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
TEMPLATE_PATH = os.path.join(BASE_DIR, "template.jpg")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error(f"Cascade file load aagala: {CASCADE_PATH}")
    st.stop()

template = cv2.imread(TEMPLATE_PATH)
if template is None:
    st.error("template.jpg repo-la illa. GitHub-ku push pannunga.")
    st.stop()

uploaded = st.file_uploader("Image select pannunga", type=["jpg", "jpeg", "png"])

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        filepath = tmp.name

    image = cv2.imread(filepath)
    if image is None:
        st.error("Invalid image")
        st.stop()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Viola-Jones
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80)
    )
    viola_result = "Face Detected" if len(faces) > 0 else "No Face"

    # 2. Template Matching
    template_result = "Not Matched"
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if (
        gray.shape[0] >= template_gray.shape[0]
        and gray.shape[1] >= template_gray.shape[1]
    ):
        res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= 0.7:
            template_result = "Matched"

    # 3 & 4. DeepFace + FaceNet
    deepface_result = "Face Not Detected"
    facenet_result = "Not Matched"

    with st.spinner("DeepFace / FaceNet run aagudhu (first time konjam neram aagum)..."):
        try:
            from deepface import DeepFace

            try:
                DeepFace.extract_faces(
                    img_path=filepath,
                    detector_backend="opencv",
                    enforce_detection=True,
                )
                deepface_result = "Face Detected"
            except Exception:
                deepface_result = "Face Not Detected"

            try:
                verify = DeepFace.verify(
                    img1_path=TEMPLATE_PATH,
                    img2_path=filepath,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=True,
                )
                facenet_result = (
                    "Face Matched" if verify["verified"] else "Not Matched"
                )
            except Exception:
                facenet_result = "Unable to Compare"
        except Exception as e:
            deepface_result = f"DeepFace load aagala: {e}"
            facenet_result = "Unable to Compare"

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Result image")

    st.subheader("Results")
    st.write(f"**Viola-Jones:** {viola_result}")
    st.write(f"**Template Matching:** {template_result}")
    st.write(f"**DeepFace:** {deepface_result}")
    st.write(f"**FaceNet:** {facenet_result}")

    os.remove(filepath)
