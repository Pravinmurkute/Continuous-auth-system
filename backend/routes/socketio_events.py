from flask_socketio import SocketIO, emit
from app import socketio

@socketio.on('connect')
def handle_connect():
    emit('response', {'message': 'Connected'})

def notify_face_detection_status(status):
    socketio.emit('face_detection_status', {'status': status})

# ...existing code...
