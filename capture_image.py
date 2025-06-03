import cv2

def save_captured_image(frame, filename):
    cv2.imwrite(f"static/captured_faces/{filename}", frame)
