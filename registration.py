from werkzeug.security import generate_password_hash
import face_recognition
import json
import cv2
import mysql.connector

def register_user(username, password):
    # ...existing code...
    hashed_password = generate_password_hash(password)
    # Save hashed_password to the database instead of the plain text password
    # ...existing code...

    # Connect to MySQL Database
    conn = mysql.connector.connect(
        host="localhost", user="root", password="yourpassword", database="yourdb"
    )
    cursor = conn.cursor()

    # Capture multiple images from webcam
    video_capture = cv2.VideoCapture(0)

    encodings = []
    for i in range(5):  # Capture 5 images
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(rgb_frame)
        
        if encoding:  
            encodings.append(encoding[0].tolist())  # Convert NumPy array to list

    video_capture.release()
    cv2.destroyAllWindows()

    # Store face encodings in database
    if encodings:
        encodings_json = json.dumps(encodings)
        user_id = 32  # Replace with the actual user ID
        cursor.execute("UPDATE users SET face_encoding=%s WHERE id=%s", (encodings_json, user_id))
        conn.commit()
        print("✅ Multiple face encodings stored successfully!")
    else:
        print("❌ No face detected. Try again.")

    conn.close()
