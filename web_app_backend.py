from flask import request, jsonify
import numpy as np
import cv2
import face_recognition

# Define known face encoding (this should be replaced with actual known encoding)
known_encoding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Example encoding

def face_authentication():
    print("Received request for face authentication")

    try:
        # Load the image from the request
        image_data = request.files.get("image")
        print("Image data received:", image_data)
        
        if image_data is None:
            print("❌ No image received in the request")
            return jsonify({"status": "fail", "reason": "No image received"})
        
        # Convert image to OpenCV format
        image = np.array(bytearray(image_data.read()), dtype=np.uint8)
        print("Image converted to byte array")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print("Image decoded to OpenCV format")

        # Detect face
        face_locations = face_recognition.face_locations(image)
        print("Face locations detected:", face_locations)
        if not face_locations:
            print("❌ No face detected in the image")
            return jsonify({"status": "fail", "reason": "No face detected"})

        # Encode face
        face_encoding = face_recognition.face_encodings(image, face_locations)
        print("Face encoding generated:", face_encoding)
        if not face_encoding:
            print("❌ Face detected but encoding failed")
            return jsonify({"status": "fail", "reason": "Encoding failed"})

        # Compare with known face
        result = face_recognition.compare_faces([known_encoding], face_encoding[0])
        print("Face Recognition Result:", result)

        if result[0]:
            print("✅ Authentication Success")
            return jsonify({"status": "success"})
        else:
            print("❌ Authentication Failed")
            return jsonify({"status": "fail", "reason": "Face did not match"})

    except Exception as e:
        print("🔥 Error in processing:", str(e))
        return jsonify({"status": "fail", "reason": str(e)})
