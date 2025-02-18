import face_recognition
import pickle
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KDTree

# Load stored face encodings
with open("models/face_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

known_encodings = np.array(known_faces["encodings"])
known_names = known_faces["names"]

# Build KDTree for fast face lookup
kdtree = KDTree(known_encodings)

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    """ Calculate Eye Aspect Ratio (EAR) for blink detection """
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def is_blinking(landmarks):
    """ Detect blinking using EAR threshold """
    left_eye = landmarks[36:42]  # Left eye landmarks
    right_eye = landmarks[42:48]  # Right eye landmarks
    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)
    avg_EAR = (left_EAR + right_EAR) / 2.0
    return avg_EAR < 0.2  # Blinking threshold

def verify_face(img):
    """ Verify face using KDTree and check for liveness (eye blink) """
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)

    if len(faces) == 0:
        print("No face detected!")
        return None

    for face in faces:
        landmarks = predictor(rgb_frame, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

        if not is_blinking(landmarks):
            print("Liveness test failed! No blink detected.")
            return None

        face_encoding = face_recognition.face_encodings(rgb_frame, [face])[0].reshape(1, -1)
        
        # KDTree lookup for the closest match
        _, indices = kdtree.query(face_encoding, k=1)
        match_index = indices[0][0]

        return known_names[match_index] if match_index in range(len(known_encodings)) else None
