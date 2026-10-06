
from flask import Flask, render_template, request
import cv2
import os
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Viola-Jones Haar Cascade
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


def template_matching(image, template):
    if template is None:
        return "Not Matched"

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Resize template if it is larger than uploaded image
    if (
        template_gray.shape[0] > image_gray.shape[0]
        or template_gray.shape[1] > image_gray.shape[1]
    ):
        scale = min(
            image_gray.shape[1] / template_gray.shape[1],
            image_gray.shape[0] / template_gray.shape[0]
        )

        new_size = (
            max(1, int(template_gray.shape[1] * scale)),
            max(1, int(template_gray.shape[0] * scale))
        )

        template_gray = cv2.resize(
            template_gray,
            new_size
        )

    result = cv2.matchTemplate(
        image_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_value, _, _ = cv2.minMaxLoc(result)

    return "Matched" if max_value >= 0.7 else "Not Matched"


def face_detection(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80)
    )

    return faces


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("image")

    if not file or file.filename == "":
        return render_template(
            "index.html",
            result="Please select an image"
        )

    filename = file.filename
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    image = cv2.imread(filepath)

    if image is None:
        return render_template(
            "index.html",
            result="Invalid image"
        )

    # --------------------------------
    # 1. Viola-Jones
    # --------------------------------

    faces = face_detection(image)

    if len(faces) > 0:
        viola_result = "Face Detected"
    else:
        viola_result = "No Face"


    # --------------------------------
    # 2. Template Matching
    # --------------------------------

    template = cv2.imread("template.jpg")

    template_result = template_matching(
        image,
        template
    )


    # --------------------------------
    # 3. DeepFace-style face detection
    # --------------------------------
    #
    # Lightweight deployment version.
    # Uses the detected face as the
    # face-detection result without
    # loading TensorFlow.

    if len(faces) > 0:
        deepface_result = "Face Detected"
    else:
        deepface_result = "Face Not Detected"


    # --------------------------------
    # 4. FaceNet-style face comparison
    # --------------------------------
    #
    # Lightweight image-feature comparison
    # so the application stays within the
    # Render Free memory limit.

    facenet_result = "Not Matched"

    if template is not None and len(faces) > 0:

        try:
            uploaded_face = image[
                faces[0][1]:faces[0][1] + faces[0][3],
                faces[0][0]:faces[0][0] + faces[0][2]
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

                uploaded_vector = (
                    uploaded_gray.flatten().astype(
                        np.float32
                    )
                )

                template_vector = (
                    template_gray.flatten().astype(
                        np.float32
                    )
                )

                uploaded_norm = np.linalg.norm(
                    uploaded_vector
                )

                template_norm = np.linalg.norm(
                    template_vector
                )

                if (
                    uploaded_norm > 0
                    and template_norm > 0
                ):

                    similarity = (
                        np.dot(
                            uploaded_vector,
                            template_vector
                        )
                        /
                        (
                            uploaded_norm
                            * template_norm
                        )
                    )

                    if similarity >= 0.80:
                        facenet_result = "Face Matched"

        except Exception:
            facenet_result = "Not Matched"


    # --------------------------------
    # Draw face rectangle
    # --------------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imwrite(
        filepath,
        image
    )


    return render_template(
        "index.html",
        viola=viola_result,
        template=template_result,
        deepface=deepface_result,
        facenet=facenet_result,
        image=filepath
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )

