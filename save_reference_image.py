import cv2

def save_reference_image():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    if ret:
        cv2.imwrite("authorized_user.jpg", frame)  # Save the image
        print("Reference image saved as 'authorized_user.jpg'")

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_reference_image()
