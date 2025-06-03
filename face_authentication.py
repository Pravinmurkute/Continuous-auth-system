import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os

# Load PyTorch-based MTCNN for face detection
face_detector = MTCNN(keep_all=True)

# Load pre-trained FaceNet model (VGGFace2 dataset)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Extract face from image
def extract_face(image_path):
    img = Image.open(image_path)
    boxes, _ = face_detector.detect(img)

    if boxes is None or len(boxes) == 0:
        return None  # No face detected

    x, y, w, h = map(int, boxes[0])
    face = img.crop((x, y, x + w, y + h))
    return face

# Get face embedding
def get_face_embedding(image_path):
    face = extract_face(image_path)
    if face is None:
        return None  # No face detected

    img_tensor = transform(face).unsqueeze(0)

    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.numpy()

# Compare two faces
def verify_face(img1, img2, threshold=0.6):
    emb1 = get_face_embedding(img1)
    emb2 = get_face_embedding(img2)

    if emb1 is None or emb2 is None:
        return False  # No face detected

    distance = np.linalg.norm(emb1 - emb2)
    return distance < threshold  # Return True if faces match

def load_stored_face_encoding(file_path):
    try:
        with open(file_path, 'r') as file:
            encoding = np.array(list(map(float, file.read().split(','))))
        return encoding
    except Exception as e:
        print(f"Error loading stored face encoding: {e}")
        return None

def save_face_encoding(encoding, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(','.join(map(str, encoding)))
        print(f"Face encoding saved to {file_path}")
    except Exception as e:
        print(f"Error saving face encoding: {e}")

if __name__ == "__main__":
    # Load the reference image and get the face embedding
    ref_image_path = "authorized_user.jpg"
    ref_embedding = get_face_embedding(ref_image_path)
    if ref_embedding is not None:
        save_face_encoding(ref_embedding, "stored_face_encoding.txt")

        # Capture a frame from the webcam
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()

        if ret:
            # Save the captured frame to a temporary file
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)

            # Get embedding for the captured frame
            face_embedding = get_face_embedding(temp_frame_path)

            # Print the encoding shapes
            print("Stored encoding shape:", ref_embedding.shape)
            print("Detected encoding shape:", face_embedding.shape)

            # Compare the captured face embedding with the reference embedding
            if verify_face(ref_image_path, temp_frame_path):
                print("Authentication Status: success")
            else:
                print("Authentication Status: error")

            # Remove the temporary file
            os.remove(temp_frame_path)

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("No face detected in the reference image.")
