from flask import Flask, render_template, request
import cv2
import os
from deepface import DeepFace

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    if not file:
        return render_template("index.html", result="Please select an image")

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    image = cv2.imread(filepath)

    # -------------------------------
    # 1. Viola-Jones
    # -------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(80, 80)
)

    viola_result = "Face Detected" if len(faces) > 0 else "No Face"


    # -------------------------------
    # 2. Template Matching
    # -------------------------------
    template = cv2.imread("template.jpg")

    template_result = "Not Matched"

    if template is not None:
        template_gray = cv2.cvtColor(
            template,
            cv2.COLOR_BGR2GRAY
        )

        if (
            gray.shape[0] >= template_gray.shape[0]
            and gray.shape[1] >= template_gray.shape[1]
        ):
            result = cv2.matchTemplate(
                gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val >= 0.7:
                template_result = "Matched"


    # -------------------------------
    # 3. DeepFace
    # -------------------------------
    deepface_result = "Face Not Detected"

    try:
        DeepFace.extract_faces(
            img_path=filepath,
            detector_backend="opencv",
            enforce_detection=True
        )

        deepface_result = "Face Detected"

    except Exception:
        deepface_result = "Face Not Detected"


    # -------------------------------
    # 4. FaceNet
    # -------------------------------
    facenet_result = "Not Matched"

    try:
        verify = DeepFace.verify(
            img1_path="template.jpg",
            img2_path=filepath,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )

        if verify["verified"]:
            facenet_result = "Face Matched"

    except Exception:
        facenet_result = "Unable to Compare"


    # Draw Viola-Jones face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imwrite(filepath, image)


    return render_template(
        "index.html",
        viola=viola_result,
        template=template_result,
        deepface=deepface_result,
        facenet=facenet_result,
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)