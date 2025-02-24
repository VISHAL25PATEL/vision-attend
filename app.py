import os
import cv2
import threading
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
from deepface import DeepFace
from scipy.spatial.distance import cosine
import zipfile
from flask import send_from_directory

# Flask app initialization
app = Flask(__name__)

# Constants
UPLOAD_FOLDER = "uploads"
KNOWN_FACES_FOLDER = "known_faces"
ATTENDANCE_CSV = "attendance.csv"
RECOGNITION_MODEL = "Facenet"
THRESHOLD = 0.4
FRAME_SKIP = 5  # Process every 5th frame

# Global variables
known_faces = {}  # Stores face embeddings
attendance = {}  # Attendance tracking
recognized_person = "Waiting..."
stop_attendance_flag = False

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------- 1️⃣ Route: Home (Upload Page) ----------
@app.route("/")
def home():
    return render_template("upload.html")


# ---------- 2️⃣ Route: Upload ZIP ----------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Extract ZIP file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(KNOWN_FACES_FOLDER)

        os.remove(file_path)  # Remove ZIP after extraction

        # Load known faces
        load_known_faces()

        return redirect(url_for("attendance_page"))


# ---------- 3️⃣ Route: Attendance Page ----------
@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


# ---------- 4️⃣ Route: Video Streaming ----------
# ---------- 4️⃣ Route: Video Streaming (Accepting Images from Frontend) ----------
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global recognized_person, attendance
    try:
        data = request.json['image']
        encoded_data = data.split(',')[1]  # Remove metadata from Base64
        image_bytes = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        recognized_person = "Unknown"
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            embedding = DeepFace.represent(face_roi, model_name=RECOGNITION_MODEL, enforce_detection=True)
            
            if embedding:
                face_embedding = np.array(embedding[0]["embedding"])
                best_match = None
                best_score = float("inf")

                for id_name, stored_embedding in known_faces.items():
                    distance = cosine(face_embedding, stored_embedding)
                    if distance < best_score and distance < THRESHOLD:
                        best_score = distance
                        best_match = id_name

                recognized_person = best_match if best_match else "Unknown"

                if recognized_person in known_faces:
                    attendance[recognized_person] = "Present"

        return jsonify({'message': 'Frame received', 'name': recognized_person})

    except Exception as e:
        return jsonify({'error': str(e)})



# ---------- Route: Get Detected Name (For Dynamic Updates) ----------
@app.route("/get_detected_name")
def get_detected_name():
    global recognized_person
    return jsonify({"name": recognized_person})


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route("/")
@app.route("/upload")
def upload():
    return render_template("upload.html")


# ---------- 6️⃣ Route: Start Attendance ----------
# Global flag to control video streaming
# Global flag to control when the webcam starts
attendance_started = False

@app.route("/start_attendance")
def start_attendance():
    global stop_attendance_flag, attendance_started
    stop_attendance_flag = False
    attendance_started = True  # Webcam starts only when this is True
    return jsonify({"message": "Attendance Started!"})

@app.route("/stop_attendance")
def stop_attendance():
    global stop_attendance_flag, attendance_started
    stop_attendance_flag = True
    attendance_started = False  # Stop the webcam when attendance stops
    return jsonify({"message": "Attendance Stopped! Redirecting..."})

@app.route("/video_feed")
def video_feed():
    return Response("", mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------- 8️⃣ Route: Final Page ----------
@app.route("/final")
def final_page():
    return render_template("final.html")


# ---------- 9️⃣ Route: View Attendance ----------
@app.route("/view_attendance")
def view_attendance():
    all_students = {}

    # Extract ID and Name from known_faces and set default attendance as "Absent"
    for filename in known_faces.keys():
        parts = filename.split("_", 1)
        student_id = parts[0]
        student_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        all_students[student_id] = {"ID": student_id, "Name": student_name, "Attendance": "Absent"}

    # Mark detected students as "Present"
    for detected_name in attendance.keys():
        parts = detected_name.split("_", 1)
        student_id = parts[0]
        if student_id in all_students:
            all_students[student_id]["Attendance"] = "Present"

    # Convert to DataFrame and add Sr. No.
    df = pd.DataFrame(list(all_students.values()))
    df.insert(0, "Sr. No.", range(1, 1 + len(df)))  # Add sequential Sr. No.

    # Convert to dictionary for rendering in HTML
    data = df.to_dict(orient="records")

    return render_template("view_attendance.html", data=data)


# ---------- 🔟 Route: Download Attendance ----------
@app.route("/download_attendance")
def download_attendance():
    all_students = {}

    # Extract ID and Name from known_faces and set default attendance as "Absent"
    for filename in known_faces.keys():
        parts = filename.split("_", 1)
        student_id = parts[0]
        student_name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        all_students[student_id] = {"ID": student_id, "Name": student_name, "Attendance": "Absent"}

    # Mark detected students as "Present"
    for detected_name in attendance.keys():
        parts = detected_name.split("_", 1)
        student_id = parts[0]
        if student_id in all_students:
            all_students[student_id]["Attendance"] = "Present"

    # Convert to DataFrame and add Sr. No.
    df = pd.DataFrame(list(all_students.values()))
    df.insert(0, "Sr. No.", range(1, 1 + len(df)))  # Add sequential Sr. No.

    # Save CSV file
    csv_path = os.path.join("static", ATTENDANCE_CSV)
    df.to_csv(csv_path, index=False)

    return redirect(url_for("static", filename=ATTENDANCE_CSV))


# ---------- Load Known Faces ----------
def load_known_faces():
    """Extracts known faces from images and stores embeddings."""
    global known_faces
    known_faces.clear()

    for filename in os.listdir(KNOWN_FACES_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(KNOWN_FACES_FOLDER, filename)
            try:
                embedding = DeepFace.represent(img_path, model_name=RECOGNITION_MODEL, enforce_detection=True)
                if embedding:
                    face_embedding = np.array(embedding[0]["embedding"])
                    name = os.path.splitext(filename)[0]
                    known_faces[name] = face_embedding
            except:
                pass


# Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Use port 10000
