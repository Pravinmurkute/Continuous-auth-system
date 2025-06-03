navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    document.querySelector("video").srcObject = stream;
})
.catch(error => console.error("Camera error:", error));
