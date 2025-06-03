import dlib
import numpy as np
import cv2

class FaceVerifier:
    def __init__(self):
        """Load DLIB face model and landmarks detector."""
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

    def get_face_embedding(self, face):
        """Generate 128D embedding for face verification."""
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        dets = self.detector(face_gray)

        for d in dets:
            shape = self.sp(face_gray, d)
            face_descriptor = self.facerec.compute_face_descriptor(face, shape)
            return np.array(face_descriptor)

        return None

    def verify_face(self, face1, face2, threshold=0.6):
        """Compare two faces using Euclidean distance."""
        distance = np.linalg.norm(face1 - face2)
        return distance < threshold
