import cv2
import face_recognition
import numpy as np
from flask import Flask, render_template, Response, jsonify
import psycopg2
import logging
import threading

app = Flask(__name__)

# Establish a connection to the database
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="your_db_name",  # Replace with your actual database name
            user="your_db_user",  # Replace with your actual database user
            password="your_db_password",  # Replace with your actual database password
            host="your_db_host",  # Replace with your actual database host
            port=5432  # Replace with your actual database port (ensure it's an integer)
        )
        return conn
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        return None

# Load known face encodings from the database
def load_known_faces():
    known_encodings = []
    known_names = []
    try:
        with get_db_connection() as conn:
            if conn is None:
                return known_encodings, known_names
            with conn.cursor() as cursor:
                cursor.execute("SELECT face_encoding, username FROM users")
                users = cursor.fetchall()
                for user in users:
                    encoding = np.array([float(num) for num in user["face_encoding"].split(",")])
                    known_encodings.append(encoding)
                    known_names.append(user["username"])
    except Exception as e:
        logging.error(f"Error loading known faces: {e}")
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

auth_status = "Unknown"

logging.basicConfig(level=logging.DEBUG)

# Load OpenCV's DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_face(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    face_locations = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Only detect faces with confidence > 50%
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x, y, x1, y1) = box.astype("int")
            face_locations.append((y, x1, y1, x))
    print("Checking if face is detected...")
    if face_locations:
        print("Face detected successfully!")
        logging.debug(f"Face detected: {face_locations}")
    else:
        print("No face detected!")
        logging.debug("No face detected.")
    return face_locations

def recognize_face(frame):
    frame_resized = cv2.resize(frame, (320, 240))  # Reduce resolution
    face_locations = detect_face(frame_resized)
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print("Faces detected:", len(face_locations))  # Debug print

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]
            auth_status = "Authenticated"
        else:
            auth_status = "Unknown"
            timestamp = cv2.getTickCount() / cv2.getTickFrequency()
            logging.warning(f"Unknown face detected at {timestamp}")

        # Draw a green rectangle around the face
        color = (0, 255, 0) if auth_status == "Authenticated" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display name above the rectangle
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return auth_status, face_locations

def generate_frames():
    global auth_status
    cap = cv2.VideoCapture(0)
    
    # Enable auto-focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # Skip initial frames for camera warm-up
    for _ in range(10):
        cap.read()

    def detect_faces():
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logging.warning("Failed to capture frame, reinitializing camera")
                cap.release()
                cap.open(0)
                continue
            
            auth_status, face_locations = recognize_face(frame)

            # Encode frame for Flask
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    thread = threading.Thread(target=detect_faces)
    thread.start()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/auth_status')
def get_auth_status():
    global auth_status
    return jsonify({"status": auth_status})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
