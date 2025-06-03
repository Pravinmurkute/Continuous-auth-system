import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        """Initialize MediaPipe face detection."""
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.6)

    def detect_face(self, frame):
        """Detect face and extract region."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = (int(bbox.xmin * w), int(bbox.ymin * h),
                                      int(bbox.width * w), int(bbox.height * h))

                face = frame[y:y + h_box, x:x + w_box]
                return face, (x, y, w_box, h_box)

        return None, None
