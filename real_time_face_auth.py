import cv2
import torch
import face_recognition
import numpy as np

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load reference image (the authorized user)
ref_image = face_recognition.load_image_file("authorized_user.jpg")  # Add your image
ref_encoding = face_recognition.face_encodings(ref_image)[0]

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = frame[:, :, ::-1]

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with reference encoding
        match = face_recognition.compare_faces([ref_encoding], face_encoding, tolerance=0.5)
        label = "Authorized" if match[0] else "Unauthorized"

        # Draw rectangle & label
        top, right, bottom, left = face_location
        color = (0, 255, 0) if match[0] else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output
    cv2.imshow("Face Authentication", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
