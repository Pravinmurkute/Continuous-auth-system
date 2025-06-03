from flask import Flask, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from camera import camera, reinitialize_camera
from face_detection import detect_face

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per minute"]
)

@app.route('/check_face_detection', methods=['GET'])
@limiter.limit("30 per minute")  # Increase to 30 requests per minute
def check_face_detection_endpoint():
    try:
        # Log the incoming request
        logging.info("Received request for /check_face_detection")

        # Capture a frame from the camera
        success, frame = camera.read()
        if not success:
            logging.warning("Failed to capture frame, reinitializing camera")
            reinitialize_camera()
            success, frame = camera.read()
            if not success:
                return jsonify({"status": "error", "message": "Failed to capture frame"}), 500

        # Call face detection function
        frame_with_faces, faces_detected = detect_face(frame)

        if faces_detected:
            return jsonify({"status": "success", "message": "Face detected"}), 200
        else:
            return jsonify({"status": "fail", "message": "No face detected"}), 404
    except Exception as e:
        logging.error(f"Error in /check_face_detection: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
