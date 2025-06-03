import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Camera utility functions
def initialize_camera(video_source=0):
    logging.info(f"Initializing camera with index {video_source}")
    cap = cv2.VideoCapture(video_source, cv2.CAP_MSMF)
    if not cap.isOpened():
        logging.error(f"❌ Failed to open camera with index {video_source} using CAP_MSMF")
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logging.error(f"❌ Failed to open camera with index {video_source} using CAP_DSHOW")
            raise ValueError("Unable to open video source", video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    logging.info(f"✅ Camera opened successfully with index {video_source}")
    return cap

def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()

def log_camera_error(index):
    logging.warning(f"Failed to capture frame, reinitializing camera with index {index}")

# Camera management functions
def list_available_cameras(max_index=10):
    available_cameras = []
    for index in range(max_index):
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            available_cameras.append(index)
            camera.release()
    logging.info(f"Available cameras: {available_cameras}")
    return available_cameras

def start_camera(video_source=0):
    cap = initialize_camera(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    release_camera(cap)

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Unable to capture frame")
    return frame

# Main function to capture faces
def capture_faces(video_source=0):
    cap = initialize_camera(video_source)
    while True:
        frame = capture_frame(cap)
        cv2.imshow('Face Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    release_camera(cap)

if __name__ == "__main__":
    capture_faces()
