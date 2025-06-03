function checkFaceDetection() {
    fetch("http://127.0.0.1:5000/check_face_detection")
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error("Fetch error:", error));
}

// Reduce polling frequency to every 1 second
setInterval(checkFaceDetection, 1000);
