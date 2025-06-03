let countdownTimer;
let timeLeft = 30; // Default 30 seconds

function checkFaceDetection() {
    fetch('/check_face_detection')
        .then(response => response.json())
        .then(data => {
            if (data.status === "face_detected") {
                document.getElementById("countdown-container").style.display = "none";
                clearInterval(countdownTimer); // Stop countdown
                timeLeft = 30; // Reset timer
            } else if (data.status === "no_face_detected") {
                document.getElementById("countdown-container").style.display = "block";
                startCountdown(data.countdown);
            } else if (data.status === "logout") {
                window.location.href = "/logout"; // Redirect to logout
            }
        })
        .catch(error => console.error("Error:", error));
}

// Start countdown
function startCountdown(initialTime) {
    timeLeft = initialTime;
    clearInterval(countdownTimer);
    countdownTimer = setInterval(() => {
        if (timeLeft <= 0) {
            clearInterval(countdownTimer);
        } else {
            timeLeft--;
            document.getElementById("countdown").innerText = timeLeft;
        }
    }, 1000);
}

// Check face detection every second
setInterval(checkFaceDetection, 1000);
