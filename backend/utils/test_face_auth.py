import time
import face_recognition
from face_recognition import load_image_file
from backend.utils.face_recognition import verify_face  # Import optimized function

# Load test image
test_img = load_image_file("dataset/test_user.jpg")

# Measure execution time
start_time = time.time()
result = verify_face(test_img)
execution_time = time.time() - start_time

if result:
    print(f"✅ Face recognized: {result}")
else:
    print("❌ Face not recognized!")

print(f"Execution Time: {execution_time:.4f} seconds")  # Print execution time
