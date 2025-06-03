import cv2
import face_recognition
import numpy as np

# Capture from webcam
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
video_capture.release()

if ret:
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face and encode
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        np.save("D:/Continuous_auth_system/stored_face.npy", face_encodings[0])  # Save encoding as .npy file
        print("✅ Face encoding stored successfully!")
    else:
        print("❌ No face detected! Try again.")
else:
    print("❌ Could not access webcam!")
