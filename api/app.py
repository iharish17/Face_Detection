from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os

app = Flask(__name__)

# Paths to models
face_proto = r"C:\Face Detection\deploy.prototxt"
face_model = r"C:\Face Detection\res10_300x300_ssd_iter_140000.caffemodel"
gender_proto = r"C:\Face Detection\gender_deploy.prototxt"
gender_model = r"C:\Face Detection\gender_net.caffemodel"

# Load networks
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Labels
GENDERS = ["Male", "Female"]

# Video capture for webcam
camera = cv2.VideoCapture(0)


def detect_faces_and_gender(frame):
    """Detect faces and classify gender on a given frame"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(cv2.resize(face, (227, 227)), 1.0,
                                              (227, 227),
                                              (78.4263377603, 87.7689143744, 114.895847746),
                                              swapRB=False)
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDERS[gender_preds[0].argmax()]

            label = f"{gender} ({confidence*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def generate_frames():
    """Generate frames from webcam for streaming"""
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = detect_faces_and_gender(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload():
    """Upload image for detection"""
    if 'image' not in request.files:
        return "No file uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result_img = detect_faces_and_gender(img)
    ret, buffer = cv2.imencode('.jpg', result_img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(debug=True)
