import cv2
from flask import Flask, Response
import requests
import face_recognition
import os
import mysql.connector
from dotenv import load_dotenv

# Face detection functions
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame, len(faces) > 0

def detect_and_display(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error loading cascade classifier")
        return

    frame, face_detected = detect_face(frame)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=50)
    cv2.imshow('Face Authentication', frame)

def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def capture_faces(video_source=0):
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Face Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Flask app for video feed
app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Change to 1 or -1 if needed

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Error: Can't read frame")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Test face recognition
def test_face_recognition():
    cap = cv2.VideoCapture(0)
    frame_count = 0

    output_dir = "d:/Continuous_auth_system/frames"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera Error")
            break

        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            print("✅ Face Detected")
        else:
            print("❌ No Face Detected")

        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

        if frame_count >= 10:  # Save 10 frames and then stop
            break

    cap.release()
    print(f"Frames saved to {output_dir}")

# Test MySQL connection
def test_db_connection():
    load_dotenv()

    db_config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "Pravin0606"),
        "database": os.getenv("MYSQL_DB", "continuous_auth"),
    }

    try:
        connection = mysql.connector.connect(**db_config)
        print("✅ MySQL Connection Successful!")
        connection.close()
    except mysql.connector.Error as err:
        print(f"❌ MySQL Connection Error: {err}")
        raise EnvironmentError("MySQL host is not configured correctly")

# Test camera with CAP_MSMF
def test_camera_msmf():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("❌ ERROR: Failed to open camera with CAP_MSMF. Check your webcam connection!")
    else:
        print("✅ Camera initialized successfully using CAP_MSMF")

    cap.release()

# Test database connection
def test_db():
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="your_username",
            password="your_password",
            database="your_database"
        )
        print("✅ Connected to MySQL successfully!")
        cursor = db.cursor()
        cursor.execute("SHOW TABLES;")
        for table in cursor.fetchall():
            print(table)
    except mysql.connector.Error as err:
        print(f"❌ MySQL Error: {err}")

if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True)
    # Uncomment the following lines to run the respective tests
    # capture_faces()
    # test_face_recognition()
    # test_db_connection()
    # test_camera_msmf()
    # test_db()
