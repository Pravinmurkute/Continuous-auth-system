# Models Directory

This directory should contain the following files:

1. `face_encodings.pkl`: A pickle file containing the known face encodings and names.
2. `shape_predictor_68_face_landmarks.dat`: A Dlib shape predictor model file for detecting facial landmarks.

## Instructions

- Ensure that both files are present in this directory before running the face recognition script.
- You can generate `face_encodings.pkl` by encoding known faces using the `face_recognition` library.
- Download `shape_predictor_68_face_landmarks.dat` from the Dlib model repository.