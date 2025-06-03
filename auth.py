from flask import Flask, render_template, Response, redirect, url_for
import cv2
import time
import face_recognition
import requests
import os
import pickle
import mysql.connector
import json
import numpy as np

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)
last_detected_time = time.time()

@app.route('/')
def index():
    return render_template('index.html')

def load_stored_encoding():
    try:
        with open("D:/Continuous_auth_system/stored_encoding.pkl", "rb") as file:
            stored_encoding, stored_face_name = pickle.load(file)
        return stored_encoding, stored_face_name
    except FileNotFoundError:
        print("Error: stored_encoding.pkl file not found.")
        return None, None

def authenticate_user(frame):
    # Load stored encoding
    stored_encoding, stored_face_name = load_stored_encoding()
    if stored_encoding is None:
        print("Authentication failed: stored encoding not found.")
        return

    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        face_location = max(face_locations, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))  # Largest face
        current_encoding = face_recognition.face_encodings(frame, [face_location])[0]
        
        face_distances = face_recognition.face_distance([stored_encoding], current_encoding)
        print(f"Face Distance: {face_distances[0]}")
        
        matches = face_recognition.compare_faces([stored_encoding], current_encoding, tolerance=0.4)  # Reduce tolerance
        if matches[0]:
            print("Authenticated as:", stored_face_name)
            print("Authentication Status: success")
        else:
            print("Authentication Failed")
    else:
        print("No face detected")

def re_register_face():
    if os.path.exists("D:/Continuous_auth_system/stored_face.jpg"):
        os.remove("D:/Continuous_auth_system/stored_face.jpg")
    else:
        print("Error: stored_face.jpg file not found.")
    # Code to capture and store a new clear image
    # ...existing code...

def detect_faces():
    global last_detected_time
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            last_detected_time = time.time()
        if time.time() - last_detected_time > 30:
            return redirect(url_for("auto_logout"))  # Redirect to logout route
        # ...existing code...

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/auto_logout')
def auto_logout():
    return "You have been logged out due to inactivity."

def main():
    # Connect to database
    conn = mysql.connector.connect(
        host="localhost", user="root", password="yourpassword", database="yourdb"
    )
    cursor = conn.cursor()

    user_id = 32  # Replace with actual user ID
    cursor.execute("SELECT face_encoding FROM users WHERE id=%s", (user_id,))
    result = cursor.fetchone()

    if result and result[0]:
        stored_encodings = json.loads(result[0])  # Convert JSON to list
    else:
        print("❌ No face encoding found for this user.")
        conn.close()
        exit()

    # Capture image for authentication
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()

    if not ret:
        print("❌ Failed to capture image.")
        conn.close()
        exit()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    live_encoding = face_recognition.face_encodings(rgb_frame)

    if not live_encoding:
        print("❌ No face detected.")
        conn.close()
        exit()

    # Compare with multiple stored encodings
    matches = face_recognition.compare_faces(np.array(stored_encodings), live_encoding[0])

    if any(matches):
        print("✅ Authentication Success!")
    else:
        print("❌ Authentication Failed!")

    conn.close()

    # Assume 'user_id' and 'auth_status' are determined by the face recognition process
    user_id = "user123"
    auth_status = "success"  # or "error"

    # Send authentication data to the backend
    response = requests.post("http://127.0.0.1:5000/authenticate", json={"user_id": user_id, "status": auth_status})

    print(response.json())  # Check response

if __name__ == '__main__':
    app.run(debug=True)
    main()
