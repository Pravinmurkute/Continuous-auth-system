import face_recognition
import cv2
import numpy as np

# Load a reference image of the authorized user
authorized_image = face_recognition.load_image_file(r"C:\Users\pravi\Pictures\Camera Roll\pravin.jpg")
authorized_encoding = face_recognition.face_encodings(authorized_image)[0]  # Get face encoding

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()
    
    # Convert frame to RGB (face_recognition requires RGB format)
    rgb_frame = frame[:, :, ::-1]
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare detected face with authorized face
        matches = face_recognition.compare_faces([authorized_encoding], face_encoding, tolerance=0.5)
        name = "Unknown"

        if matches[0]:
            name = "Authorized User"

        # Draw rectangle and label
        color = (0, 255, 0) if name == "Authorized User" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show frame
    cv2.imshow("Face Authentication", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
video_capture.release()
cv2.destroyAllWindows()
