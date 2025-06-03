var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('connect', function() {
    console.log('Connected to WebSocket');
});

socket.on('face_detection_status', function(data) {
    console.log('Face detection status:', data.status);
    // Update the UI based on the face detection status
});
