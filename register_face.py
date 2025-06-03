import cv2

def capture_face_image():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    success, frame = camera.read()
    if success:
        cv2.imwrite("D:/Continuous_auth_system/stored_face.jpg", frame)
        print("Face image saved successfully!")
    else:
        print("Failed to capture face image.")
    camera.release()

if __name__ == "__main__":
    capture_face_image()
