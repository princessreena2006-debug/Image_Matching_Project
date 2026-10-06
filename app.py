import streamlit as st
import cv2
import numpy as np

st.title("Image Matching & Face Detection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    data = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80)
    )

    # Template Matching
    template = cv2.imread("template.jpg")
    template_result = "Not Matched"

    if template is not None:

        template_gray = cv2.cvtColor(
            template,
            cv2.COLOR_BGR2GRAY
        )

        if (
            template_gray.shape[0] <= gray.shape[0]
            and template_gray.shape[1] <= gray.shape[1]
        ):

            result = cv2.matchTemplate(
                gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_value, _, _ = cv2.minMaxLoc(result)

            if max_value >= 0.7:
                template_result = "Matched"

    # Viola-Jones
    if len(faces) > 0:
        viola_result = "Face Detected"
    else:
        viola_result = "No Face"

    # DeepFace-style result
    if len(faces) > 0:
        deepface_result = "Face Detected"
    else:
        deepface_result = "Face Not Detected"

    # FaceNet-style comparison
    facenet_result = "Not Matched"

    if template is not None and len(faces) > 0:

        x, y, w, h = faces[0]

        uploaded_face = image[
            y:y + h,
            x:x + w
        ]

        if uploaded_face.size > 0:

            uploaded_face = cv2.resize(
                uploaded_face,
                (128, 128)
            )

            template_resized = cv2.resize(
                template,
                (128, 128)
            )

            uploaded_gray = cv2.cvtColor(
                uploaded_face,
                cv2.COLOR_BGR2GRAY
            )

            template_gray = cv2.cvtColor(
                template_resized,
                cv2.COLOR_BGR2GRAY
            )

            uploaded_vector = uploaded_gray.flatten().astype(
                np.float32
            )

            template_vector = template_gray.flatten().astype(
                np.float32
            )

            uploaded_norm = np.linalg.norm(
                uploaded_vector
            )

            template_norm = np.linalg.norm(
                template_vector
            )

            if uploaded_norm > 0 and template_norm > 0:

                similarity = (
                    np.dot(
                        uploaded_vector,
                        template_vector
                    )
                    /
                    (
                        uploaded_norm *
                        template_norm
                    )
                )

                if similarity >= 0.80:
                    facenet_result = "Face Matched"

    # Draw face rectangle
    for x, y, w, h in faces:

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    st.image(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        ),
        caption="Uploaded Image"
    )

    st.subheader("Results")

    st.write(
        "Template Matching:",
        template_result
    )

    st.write(
        "Viola-Jones:",
        viola_result
    )

    st.write(
        "DeepFace:",
        deepface_result
    )

    st.write(
        "FaceNet:",
        facenet_result
    )
