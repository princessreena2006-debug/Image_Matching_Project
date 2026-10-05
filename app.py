```python
from flask import Flask, render_template, request
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


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

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    image = cv2.imread(filepath)

    if image is None:
        return render_template(
            "index.html",
            result="Invalid image"
        )

    # -------------------------------
    # 1. Viola-Jones
    # -------------------------------

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

    viola_result = (
        "Face Detected"
        if len(faces) > 0
        else "No Face"
    )


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
    # 3. DeepFace + FaceNet
    # -------------------------------

    deepface_result = "Face Not Detected"
    facenet_result = "Unable to Compare"

    try:

        # Lazy import to reduce startup memory
        from deepface import DeepFace

        # DeepFace detection
        DeepFace.extract_faces(
            img_path=filepath,
            detector_backend="opencv",
            enforce_detection=True
        )

        deepface_result = "Face Detected"

        # FaceNet verification
        verify = DeepFace.verify(
            img1_path="template.jpg",
            img2_path=filepath,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True
        )

        if verify["verified"]:
            facenet_result = "Face Matched"
        else:
            facenet_result = "Not Matched"

    except Exception as e:

        deepface_result = "Face Not Detected"
        facenet_result = "Unable to Compare"


    # -------------------------------
    # Draw Viola-Jones rectangle
    # -------------------------------

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
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
```
