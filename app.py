import os
import pymysql
import logging
import socket
import cv2
import numpy as np
import base64
#import face_recognition # Keep for potential use in update_profile or remove if fully replaced
from datetime import timedelta, datetime
from dotenv import load_dotenv
from flask import (
    Flask, render_template, session, redirect, url_for, request,
    jsonify, Response, flash, send_from_directory, abort, make_response
)
from flask_session import Session
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, SubmitField, FileField, HiddenField
from wtforms.validators import DataRequired, Regexp, Email # Added Email validator
from werkzeug.security import generate_password_hash, check_password_hash
# import mysql.connector  # type: ignore # Using pymysql primarily now
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from backend.utils.config import Config # Assuming this path is correct
import redis
# from flask_mysqldb import MySQL # Removed, using pymysql
import re
import warnings
# from flask_socketio import SocketIO # SocketIO setup not completed, commented out for now
import time
from scipy.spatial.distance import cosine # Used if comparing embeddings with cosine distance
import io
import threading
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
# import torchvision.models as models # Not used directly?
from PIL import Image
from flask_compress import Compress
from waitress import serve
import mediapipe as mp
import validators  # Added for URL validation

# --- Environment and Path Setup ---
from functools import wraps
load_dotenv()

# 🛠️ Ensure Python can find the 'auth' package
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))) # Look in current dir too

# ✅ Import Face and Voice Modules Properly
try:
    # Add your logic here
    pass
except Exception as e:
    logging.error(f"An error occurred: {e}")
    pass  # Add your logic here
except Exception as e:
    logging.error(f"An error occurred: {e}")
    # Add your code here
    pass
except Exception as e:
    logging.error(f"An error occurred: {e}")
    from auth.face_verification import FaceVerifier
    from auth.voice_authentication import VoiceAuthenticator
    from auth.face_detection import FaceDetector
    logging.info("Successfully imported auth modules.")
except ImportError as e:
    logging.error(f"ImportError: {e}. Ensure the 'auth' module and submodules exist.")
    # Depending on severity, you might raise the error or continue with reduced functionality
    # raise e  # Uncomment to make this a fatal error
    # Initialize dummy classes if auth module is optional/missing
    class FaceVerifier: pass
    class VoiceAuthenticator: pass
    class FaceDetector: pass
    logging.warning("Auth modules not found, continuing without face/voice auth features from module.")


# --- Logging Configuration ---
# Set logging level to WARNING to hide DEBUG and INFO messages for some libraries
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress OpenCV warnings/errors (use only one method)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
# if hasattr(cv2.utils, "logging") and hasattr(cv2.utils.logging, "setLogLevel"):
#     cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
# else:
#     logging.warning("⚠️ Logging control not supported in this OpenCV version.")

# Ignore specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning) # General future warnings


# --- Environment Variable Validation ---
required_env_vars = [
    "SECRET_KEY", "MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DB",
    "WTF_CSRF_SECRET_KEY", "REDIS_URL"
]
for var in required_env_vars:
    if not os.getenv(var):
        logging.error(f"Missing environment variable: {var}")
        raise EnvironmentError(f"Missing environment variable: {var}")

MYSQL_HOST = os.getenv("MYSQL_HOST")

# --- Initial Checks (Database, etc.) ---
def is_mysql_server_reachable(host, port=3306, user=None, password=None, db=None):
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user or os.getenv("MYSQL_USER"),
            password=password or os.getenv("MYSQL_PASSWORD"),
            database=db or os.getenv("MYSQL_DB"),
            connect_timeout=5
        )
        conn.close()
        logging.info(f"✅ MySQL server connection successful at {host}:{port}")
        return True
    except pymysql.MySQLError as e:
        logging.error(f"❌ MySQL server connection failed at {host}:{port}. Error: {e}")
        return False

if not is_mysql_server_reachable(MYSQL_HOST):
     
    raise EnvironmentError(f"Initial MySQL server check failed for {MYSQL_HOST}. Please ensure it's running and accessible.")


# --- Flask App Initialization and Configuration ---
app = Flask(__name__, static_folder="static")
app.config.from_object(Config)
app.secret_key = os.getenv("SECRET_KEY") # Needed for session and flash messages

# CSRF Protection
app.config["WTF_CSRF_ENABLED"] = True
app.config["WTF_CSRF_SECRET_KEY"] = os.getenv("WTF_CSRF_SECRET_KEY")
csrf = CSRFProtect(app)

# Session Configuration (Using Flask-Session with Redis) # MODIFIED COMMENT
app.config['SESSION_TYPE'] = 'redis'  # CHANGED to redis
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_NAME'] = 'auth_session_redis'  # CHANGED name (recommended)
app.config['SESSION_COOKIE_SECURE'] = False  # Use False for local HTTP development
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# --- ADDED Redis connection for Flask-Session ---
# Assumes REDIS_URL is like 'redis://localhost:6379/0' in .env
app.config['SESSION_REDIS'] = redis.from_url(os.getenv("REDIS_URL"))
# -------------------------------------------------

# Re-initialize Session AFTER setting the config
Session(app)

# Response Compression
Compress(app)

# Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=app.config["REDIS_URL"],
    default_limits=["200 per day", "50 per hour", "10 per minute"] # Example limits
)

# --- Database Connection ---
db_config = {
    'host': os.getenv("MYSQL_HOST"),
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASSWORD"),
    'database': os.getenv("MYSQL_DB"),
    'cursorclass': pymysql.cursors.DictCursor, # Return rows as dictionaries
    'connect_timeout': 10,
    'charset': 'utf8mb4'
}

def get_db_connection():
    try:
        # Consider using a connection pool for better performance in production
        return pymysql.connect(**db_config)
    except pymysql.MySQLError as e:
        logging.error(f"Database connection failed: {e}")
        return None # Or raise an exception

# --- Face Recognition Model Setup ---
# Load pre-trained FaceNet model once at startup
try:
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    logging.info("FaceNet model loaded successfully.")
    # Image preprocessing
    img_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize for FaceNet
    ])
except Exception as e:
    logging.error(f"Failed to load FaceNet model: {e}")
    facenet_model = None # Handle model loading failure gracefully


# --- Global State and Locks (for Threading) ---
auth_status = {'authenticated': False, 'user_id': None, 'last_check': time.time()} # Thread-safe state dictionary
last_face_detected_time = time.time()
cap = None # Camera object
camera_lock = threading.Lock()
auth_status_lock = threading.Lock()
# session_lock = threading.Lock() # Generally avoid locking session, manage state via auth_status

# --- Camera Initialization ---
def initialize_camera(index=0):
    global cap
    with camera_lock:
        if cap and cap.isOpened():
            cap.release()
        logging.info(f"Attempting to initialize camera with index {index} using CAP_DSHOW...")
        cap_temp = cv2.VideoCapture(index, cv2.CAP_DSHOW) # Try DirectShow first
        if not cap_temp.isOpened():
            logging.warning(f"Failed with CAP_DSHOW, trying default backend for index {index}...")
            cap_temp = cv2.VideoCapture(index) # Try default
        if cap_temp.isOpened():
            cap = cap_temp
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Optional: Set resolution
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logging.info(f"✅ Camera {index} initialized successfully.")
            return True
        else:
            cap = None
            logging.error(f"❌ Failed to initialize camera with index {index}.")
            return False

def release_camera():
    global cap
    with camera_lock:
        if cap and cap.isOpened():
            cap.release()
            cap = None
            logging.info("Camera released.")

# Initialize camera at startup
initialize_camera()

# --- Face Detection and Verification Helpers ---
# Using OpenCV's Cascade Classifier for basic detection in background thread
# --- Initialize MediaPipe Face Detection ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils  # Optional: For drawing boxes
face_detector_mp = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
logging.info("✅ MediaPipe Face Detector initialized.")

def detect_faces_mediapipe(frame):
    """
    Detects faces using MediaPipe Face Detection.

    Args:
        frame: BGR image (NumPy array) from OpenCV.

    Returns:
        A list of bounding boxes, where each box is [x, y, w, h].
        Returns an empty list if no faces are detected or on error.
    """
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # Improve performance
        results = face_detector_mp.process(image_rgb)
        image_rgb.flags.writeable = True

        detected_faces = []
        if results.detections:
            img_h, img_w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * img_w)
                y = int(bboxC.ymin * img_h)
                w = int(bboxC.width * img_w)
                h = int(bboxC.height * img_h)
                x, y, w, h = max(0, x), max(0, y), min(img_w - x, w), min(img_h - y, h)
                detected_faces.append([x, y, w, h])
        return detected_faces
    except Exception as e:
        logging.error(f"Error during MediaPipe face detection: {e}", exc_info=True)
        return []

def extract_face_pil(image_data):
    """Extracts face using PIL, assumes image_data is bytes or readable path."""
    try:
        if isinstance(image_data, np.ndarray):
             # Convert BGR (OpenCV) to RGB (PIL)
            img = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        elif isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        else: # Assume path
             img = Image.open(image_data).convert("RGB")
        return img
    except Exception as e:
        logging.error(f"Error converting image data to PIL Image: {e}")
        return None

def get_face_embedding(face_image_pil):
    """Generates embedding from a PIL image."""
    if facenet_model is None or face_image_pil is None:
        logging.warning("FaceNet model not loaded or no face image provided.")
        return None
    try:
        img_tensor = img_transform(face_image_pil).unsqueeze(0) # Apply transforms and add batch dim
        with torch.no_grad():
            embedding = facenet_model(img_tensor)
        return embedding.numpy().flatten() # Return as 1D numpy array
    except Exception as e:
        logging.error(f"Error generating face embedding: {e}")
        return None

def verify_face(embedding1, embedding2, threshold=1.1): # INCREASED THRESHOLD (e.g., to 1.0, 1.1, or 1.2)
    """Compares two embeddings using Euclidean distance."""
    if embedding1 is None or embedding2 is None:
        return False
    distance = np.linalg.norm(embedding1 - embedding2)
    logging.debug(f"Face verification distance: {distance} (Threshold: {threshold})")
    return distance < threshold

# --- WTForms Definitions ---
class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])  # Changed to 'email' with Email validator
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

class RegisterForm(FlaskForm):
    full_name = StringField("Full Name", validators=[DataRequired()])
    username = StringField("Username", validators=[
        DataRequired(),
        Regexp('^[a-zA-Z0-9]+$', message="Username must contain only letters and numbers.")
    ]) # <-- ADDED USERNAME FIELD
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Register")

class UpdatePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired()])
    submit = SubmitField('Update Password')

class UpdateProfileForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    # Use FileField for uploads, handle base64 separately if needed from JS
    face_image = FileField('New Face Image (Optional)')
    submit = SubmitField('Update Profile')


# --- Background Monitoring Threads ---

def monitor_authentication():
    """Background thread to check if user should be logged out based on auth status."""
    global auth_status
    logging.info("🔒 Authentication monitoring thread started.")
    while True:
        time.sleep(30) # Check every 30 seconds
        try:
            with auth_status_lock:
                should_logout = not auth_status.get('authenticated', False) and auth_status.get('user_id') is not None
                last_check_time = auth_status.get('last_check', time.time())

            if should_logout:
                logging.warning(f"🔒 Auth status indicates logout needed for user {auth_status.get('user_id')}. Triggering logout.")
                # How to trigger logout from thread?
                # Option 1: Set a flag another part of the app checks (less immediate)
                # Option 2: Make an internal request to a logout endpoint (complex)
                # Option 3: Directly modify auth_status (simplest for now)
                with auth_status_lock:
                     auth_status['user_id'] = None # Clear user ID
                     auth_status['authenticated'] = False # Ensure it's false
                # NOTE: This doesn't clear the Flask session directly. Logout relies
                # on subsequent requests checking auth_status or session timeout.

           


        except Exception as e:
            logging.error(f"Error in monitor_authentication thread: {e}", exc_info=True)


def monitor_face():
    """Background thread to monitor for face presence."""
    global last_face_detected_time, cap, auth_status
    logging.info("🙂 Face monitoring thread started.")
    no_face_logout_threshold = 60 # Seconds without detecting a face to trigger logout

    while True:
        time.sleep(5) # Check every 5 seconds
        frame = None
        try:
            with camera_lock:
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        logging.warning("🙂 Failed to read frame in monitor_face.")
                        frame = None
                        # Attempt to reinitialize camera if reading fails consistently?
                        # initialize_camera() # Be careful about race conditions here
                else:
                    # Attempt to initialize if not open
                    logging.warning("🙂 Camera not open in monitor_face, attempting init.")
                    initialize_camera()
                    continue # Skip this cycle

            if frame is not None:
                faces = detect_faces_mediapipe(frame) # Use the simple detector here
                if len(faces) > 0:
                    # logging.debug("🙂 Face detected.")
                    last_face_detected_time = time.time()
                else:
                    # No face detected
                    time_since_last_face = time.time() - last_face_detected_time
                    # logging.debug(f"🙂 No face detected for {time_since_last_face:.1f}s")
                    with auth_status_lock:
                        current_user_id = auth_status.get('user_id')

                    if current_user_id is not None and time_since_last_face > no_face_logout_threshold:
                        logging.warning(f"🙂 User {current_user_id} absent for > {no_face_logout_threshold}s. Triggering logout.")
                        with auth_status_lock:
                            auth_status['authenticated'] = False # Mark as unauthenticated
                            # Keep user_id for logging? Or clear it? Let's clear it.
                            auth_status['user_id'] = None
                        # Consider releasing camera if user is logged out?
                        # release_camera() # Maybe not, app might need it again soon

        except Exception as e:
            logging.error(f"Error in monitor_face thread: {e}", exc_info=True)


# --- Helper Functions (Logging, etc.) ---

def log_authentication_attempt(user_id, username, status):
    """Logs authentication success or failure."""
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to log auth attempt: No DB connection.")
        return

    # Determine event type and details based on status
    event_type = 'login_attempt'
    details = f"Login attempt for user: {username}" if username else "Login attempt with unknown username"
    if status == 'success':
        event_type = 'login_success'
        details = f"Successful login for user: {username}"
    elif status == 'fail':
        event_type = 'login_fail'
        details = f"Failed login for user: {username}"
    elif status == 'logout':
        event_type = 'logout'
        details = f"User logout: {username}"
    elif 'fail_' in status:
        event_type = 'auth_fail_continuous'
        details = f"Continuous auth check failed for user {username}: {status}"
    elif 'success_' in status:
        event_type = 'auth_success_continuous'
        details = f"Continuous auth check succeeded for user {username}: {status}"

    try:
        with conn.cursor() as cursor:
            # --- CORRECTED INSERT STATEMENT ---
            sql = """INSERT INTO authlogs (user_id, event_type, status, details, timestamp)
                     VALUES (%s, %s, %s, %s, NOW())"""
            if user_id is not None:
                try:
                    user_id = int(user_id)
                except ValueError:
                    logging.error(f"Invalid user_id '{user_id}' passed to log_authentication_attempt.")
                    user_id = None

            # Truncate status and details if needed
            status = status[:20] if status else status
            event_type = event_type[:50] if event_type else event_type
            details = details[:500] if details else details

            cursor.execute(sql, (user_id, event_type, status, details))
            conn.commit()
            logging.debug(f"Auth log inserted: User {user_id}, Event {event_type}, Status {status}")
    except pymysql.MySQLError as e:
        logging.error(f"MySQL Error logging authentication: {e}")
        conn.rollback()
    except Exception as e:
        logging.error(f"Unexpected error logging authentication: {e}")
    finally:
        if conn:
            conn.close()

def save_face_encoding(user_id, face_encoding):
    """Saves the face encoding to a file."""
    save_path = f"D:/Continuous_auth_system/user_embeddings"
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create directory if it doesn't exist
        logging.info(f"Created directory for user embeddings: {save_path}")

    file_path = f"{save_path}/{user_id}_face.npy"
    try:
        np.save(file_path, face_encoding)
        logging.info(f"✅ Face encoding saved at {file_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save face encoding for user {user_id}: {e}")

# --- Flask Routes ---

@app.before_request
def before_request_checks():
    session.permanent = True
    # pass # NO other logic for this test run

@app.route("/")
def home():
    """Serves the home/landing page."""
    return render_template("home.html")



@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        full_name = form.full_name.data
        username = form.username.data
        email = form.email.data
        password = form.password.data
        hashed_password = generate_password_hash(password)

        logging.info(f"Registration attempt: Full Name='{full_name}', Username='{username}', Email='{email}'")

        conn = get_db_connection()
        if not conn:
            flash("Database connection error.", "danger")
            return render_template('register.html', form=form)

        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT user_id FROM users WHERE email = %s OR username = %s", (email, username))
                existing_user = cursor.fetchone()
                if existing_user:
                    flash('Email address or username already exists.', "danger")
                    return render_template('register.html', form=form)

                cursor.execute(
                    "INSERT INTO users (full_name, username, email, password_hash, role) VALUES (%s, %s, %s, %s, %s)",
                    (full_name, username, email, hashed_password, 'user')
                )
                conn.commit()
                new_user_id = cursor.lastrowid
                logging.info(f"User '{full_name}' registered successfully with ID {new_user_id}.")

                # --- MODIFIED PART ---
                # Instead of redirecting to login, redirect to a new face enrollment page
                flash(f"Registration successful, {full_name}! Please enroll your face.", "success")
                return redirect(url_for('enroll_face_page', user_id=new_user_id))
                # --- END MODIFIED PART ---

        except pymysql.MySQLError as e:
            logging.error(f"Database error during registration: {e}")
            conn.rollback()
            flash('Database error occurred during registration.', "danger")
            return render_template('register.html', form=form)
        except Exception as e:
            logging.error(f"Unexpected error during registration: {e}", exc_info=True)
            if conn: conn.rollback()
            flash('An unexpected error occurred during registration.', "danger")
            return render_template('register.html', form=form)
        finally:
            if conn:
                conn.close()

    elif request.method == 'POST':
        flash("Please correct the errors below.", "warning")

    return render_template('register.html', form=form)

@app.route('/login', methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    form = LoginForm()  # Uses the LoginForm with 'email' field
    if form.validate_on_submit():
        email_attempt = form.email.data
        password = form.password.data

        conn = get_db_connection()
        user = None
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT user_id, full_name, email, password_hash, role FROM users WHERE email = %s",
                        (email_attempt,)
                    )
                    user = cursor.fetchone()
            except pymysql.MySQLError as e:
                logging.error(f"Database error during login query for email {email_attempt}: {e}")
                flash("Database error during login.", "danger")
            finally:
                conn.close()
        else:
            logging.error("Failed to get DB connection for login.")
            flash("Database connection error.", "danger")

        password_match = False
        if user:
            stored_hash = user.get("password_hash")
            if stored_hash:
                try:
                    password_match = check_password_hash(stored_hash, password)
                except Exception as check_err:
                    logging.error(f"Error during check_password_hash for email {email_attempt}: {check_err}")
            else:
                logging.warning(f"Stored password hash is missing for user email: {email_attempt}")
        else:
            logging.warning(f"Login attempt failed: User not found for email: {email_attempt}")

        if user and password_match:
            session.clear()
            session['user_id'] = user['user_id']
            session['full_name'] = user['full_name']
            session['email'] = user['email']
            session['role'] = user['role']
            session.permanent = True

            with auth_status_lock:
                auth_status['authenticated'] = True
                auth_status['user_id'] = user['user_id']
                auth_status['last_check'] = time.time()

            log_authentication_attempt(user['user_id'], user['full_name'], "success")
            logging.info(f"User '{user['full_name']}' (Email: {user['email']}) logged in successfully.")

            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            log_authentication_attempt(user['user_id'] if user else None, email_attempt, "fail")
            logging.warning(f"Failed login attempt for email: {email_attempt}")
            flash("Invalid email or password.", "danger")

    return render_template("login.html", form=form)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login to access this page.", "warning")
            return redirect(url_for('login', next=request.path))
        with auth_status_lock:
            session_user_id = session.get('user_id')
            auth_user_id = auth_status.get('user_id')
            is_authenticated = auth_status.get('authenticated', False)
            if not is_authenticated or (session_user_id is not None and session_user_id != auth_user_id):
                logging.warning(f"Access denied for session user {session_user_id}. Auth Status: {auth_status}")
                session.clear()
                if session_user_id == auth_user_id:
                    auth_status['authenticated'] = False
                    auth_status['user_id'] = None
                return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/dashboard")
@login_required
def dashboard():
    """Handles the regular user dashboard view."""
    user_display_name = session.get("full_name", "Unknown User")
    return render_template("dashboard.html", full_name=user_display_name)

@app.route("/logout")
def logout():
    full_name = session.get('full_name', 'unknown')
    user_id = session.get('user_id')
    if user_id:
        log_authentication_attempt(user_id, full_name, "logout")

    # Clear Flask session
    session.clear()

    # Update global auth status if this user was the one tracked
    with auth_status_lock:
        # Check if the logged-out user_id matches the one in auth_status
        # Avoid clearing status if a different user logged in meanwhile (unlikely but possible)
        if 'user_id' in auth_status and auth_status['user_id'] == user_id:
             auth_status['authenticated'] = False
             auth_status['user_id'] = None
             auth_status['last_check'] = time.time() # Update last check time

    flash("You have been logged out successfully.", "success")
    logging.info(f"User {full_name} (ID: {user_id}) logged out.")
    return redirect(url_for('home'))

# --- Continuous Authentication Routes ---

@app.route('/video_feed')
def video_feed():
    """Provides the video stream from the camera."""
    if 'user_id' not in session:
        abort(401)  # Unauthorized if not logged in
    current_full_name = session.get("full_name", "Unknown")  # Get username once
    # -----------------------------------------------------------------

    def generate_frames(full_name_to_display):  # Pass username as argument
        while True:
            frame = None
            with camera_lock:
                if cap and cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        frame = None
                if frame is None:
                    # Create a black frame or a "camera unavailable" image
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera Error", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    time.sleep(1)  # Avoid busy-looping if camera is dead
                    continue

            # --- Use the passed argument ---
            cv2.putText(frame, f"User: {full_name_to_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # -----------------------------

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Adjust quality
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)  # Limit frame rate slightly

    # --- Pass the username when calling the generator ---
    return Response(generate_frames(current_full_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_face_authentication', methods=['POST'])
@limiter.limit("10 per minute") # Limit verification checks
def check_face_authentication():
    """Endpoint called periodically by JS to verify the current face."""
    # --- Use .get() for session access ---
    user_id = session.get('user_id')     # <--- CORRECTED
    full_name = session.get('full_name')   # <--- CORRECTED
    # ------------------------------------

    if not user_id:
        # ... handle missing session ...+
        return jsonify({'status': 'error', 'message': 'Session expired or invalid. Please login again.'}), 401

    # --- Load Stored Face Encoding ---
    # IMPORTANT: This needs to load the specific user's encoding!
    # Placeholder: Load a generic one. In reality, fetch from DB based on user_id.
    stored_encoding = None
    stored_encoding_path = f"D:/Continuous_auth_system/user_embeddings/{user_id}_face.npy" # Example path structure
  

    # Using file path for now:
    try:
        if os.path.exists(stored_encoding_path):
            stored_encoding = np.load(stored_encoding_path)
            logging.debug(f"Loaded stored encoding for user {user_id}")
        else:
             logging.warning(f"Stored face encoding file not found for user {user_id} at {stored_encoding_path}")
             # Decide how to handle missing enrollment: fail, prompt enrollment?
             return jsonify({'status': 'error', 'message': 'Face enrollment not found for user.'}), 404
    except Exception as e:
        logging.error(f"Error loading stored encoding for user {user_id}: {e}")
        return jsonify({'status': 'error', 'message': 'Could not load stored face data.'}), 500

    # --- Capture Current Face ---
    frame = None
    with camera_lock:
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame = None
        else:
             initialize_camera() # Try to reinit if not open

    if frame is None:
        logging.warning(f"Failed to capture frame for face auth check (User: {full_name})")
        return jsonify({'status': 'error', 'message': 'Camera issue during verification'}), 500

    # --- Process and Compare ---
    face_image_pil = extract_face_pil(frame)
    if face_image_pil is None:
        # This indicates no face was detected by the extraction/conversion
        log_authentication_attempt(user_id, full_name, "fail_no_face")
        with auth_status_lock: # Update global state if no face detected during check
            if auth_status.get('user_id') == user_id:
                 auth_status['authenticated'] = False # Mark as unauth
        return jsonify({'status': 'fail', 'authenticated': False, 'message': 'No face detected in frame.'}), 200 # 200 OK, but auth failed

    current_embedding = get_face_embedding(face_image_pil)
    if current_embedding is None:
        # Model failed or detected face was unsuitable
        logging.warning(f"Could not generate embedding from current frame (User: {full_name})")
        # Don't necessarily fail auth here, could be transient issue
        return jsonify({'status': 'error', 'message': 'Could not process face image.'}), 500

    # --- Verification ---
    match = verify_face(current_embedding, stored_encoding)
    log_status = "success_face" if match else "fail_face"
    log_authentication_attempt(user_id, full_name, log_status)

    with auth_status_lock:
        auth_status['authenticated'] = match # Update global status based on check
        auth_status['user_id'] = user_id if match else None # Keep user_id if matched
        auth_status['last_check'] = time.time()

    return jsonify({'status': 'success' if bool(match) else 'fail', 'authenticated': bool(match)})

# --- User Profile and Settings Routes ---

@app.route("/userprofile")
def userprofile():
    if 'user_id' not in session:
        flash("Please login first.", "info")
        return redirect(url_for('login')) # Redirect if not logged in

    user_id = session['user_id']
    conn = get_db_connection()
    user = None
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT user_id, full_name, email, role FROM users WHERE user_id = %s", (user_id,))
                user = cursor.fetchone()
        # --- ADDED except ---
        except pymysql.MySQLError as e:
            logging.error(f"DB error fetching profile for user {user_id}: {e}")
            flash("Could not load profile due to a database error.", "danger")
        # --- ADDED finally ---
        finally:
            conn.close() # Close connection

    if user:
        return render_template('userprofile.html', user=user)
    else:
        # If user is None but no DB error occurred, means user ID from session doesn't exist
        if not flash: # Only flash 'not found' if DB error didn't already flash
             flash("User not found.", "danger")
        session.clear() # Log out if user doesn't exist in DB
        return redirect(url_for('login'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'user_id' not in session:
        flash("Please login first.", "info")
        return redirect(url_for('login')) # Redirect if not logged in

    user_id = session['user_id']
    form = UpdateProfileForm()
    conn = None # Define conn outside try

    if form.validate_on_submit():
        new_email = form.email.data
        face_image_file = form.face_image.data

        conn = get_db_connection() # Get connection for POST handling
        if not conn:
             flash("Database error, cannot update settings.", "danger")
             # Don't redirect here, re-render the form with the error
             return render_template('settings.html', form=form, current_email=session.get('email'))

        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT user_id FROM users WHERE email = %s AND user_id != %s", (new_email, user_id))
                if cursor.fetchone():
                    flash("Email address is already in use.", "danger")
                    # Re-render form, don't proceed
                    return render_template('settings.html', form=form, current_email=session.get('email'))

                cursor.execute("UPDATE users SET email = %s WHERE user_id = %s", (new_email, user_id))
                session['email'] = new_email
                flash("Email updated successfully!", "success")

                # Handle face image update
                if face_image_file:
                    try:
                        image_bytes = face_image_file.read()
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        new_embedding = get_face_embedding(pil_image)
                        if new_embedding is not None:
                            save_face_encoding(user_id, new_embedding)
                            logging.info(f"New face embedding stored for user {user_id}.")
                            flash("Face profile updated successfully!", "success")
                        else:
                            flash("Could not process the uploaded face image.", "warning")
                    except Exception as e:
                         logging.error(f"Error processing uploaded face image for user {user_id}: {e}")
                         flash("Error updating face profile.", "danger")
                     # --- Make sure commit happens AFTER potential face processing ---
                conn.commit() # Commit changes (email update and potentially face encoding in DB if added)
                return redirect(url_for('settings')) # Redirect after successful POST and commit
        # --- ADDED except ---
        except pymysql.MySQLError as e:
            logging.error(f"DB error updating settings for user {user_id}: {e}")
            if conn: conn.rollback() # Rollback on error
            flash("Database error saving settings.", "danger")
        # --- ADDED finally ---
        finally:
             if conn: conn.close() # Close connection

    # --- GET Request Handling ---
    if request.method == 'GET':
        conn_get = get_db_connection() # Use separate var for GET conn
        if conn_get:
            try:
                with conn_get.cursor() as cursor:
                     cursor.execute("SELECT email FROM users WHERE user_id = %s", (user_id,))
                     user_data = cursor.fetchone()
                     if user_data:
                         # Pre-populate the form for GET request
                         form.email.data = user_data['email']
            # --- ADDED except ---
            except pymysql.MySQLError as e:
                 logging.error(f"DB error fetching current settings for user {user_id}: {e}")
                 flash("Could not load current settings.", "warning")
            # --- ADDED finally ---
            finally:
                 if conn_get: conn_get.close() # Close GET connection

    # Render the template for GET requests or failed POST validation
    return render_template('settings.html', form=form, current_email=session.get('email'))

@app.route("/update_password", methods=["GET", "POST"])
def update_password():
    if "user_id" not in session:
        flash("Please login first.", "info")
        return redirect(url_for("login"))

    user_id = session['user_id']
    form = UpdatePasswordForm()
    if form.validate_on_submit():
        current_password = form.current_password.data
        new_password = form.new_password.data

        conn = get_db_connection()
        if not conn:
            flash("Database error.", "danger")
            return render_template("update_password.html", form=form)

        try:
            user = None # Define user before with block
            with conn.cursor() as cursor:
                cursor.execute("SELECT password_hash FROM users WHERE user_id = %s", (user_id,))
                user = cursor.fetchone()
                if not user or not check_password_hash(user["password_hash"], current_password):
                    flash("Invalid current password.", "danger")
                elif current_password == new_password:
                     flash("New password cannot be the same as the current password.", "warning")
                else:
                    hashed_new_password = generate_password_hash(new_password)
                    cursor.execute("UPDATE users SET password_hash = %s WHERE user_id = %s", (hashed_new_password, user_id))
                    conn.commit() # Commit the change
                    flash("Password updated successfully.", "success")
                    logging.info(f"Password updated for user {user_id}.")
                    return redirect(url_for('settings')) # Redirect after success
        # --- ADDED except ---
        except pymysql.MySQLError as e:
             logging.error(f"DB error updating password for user {user_id}: {e}")
             if conn: conn.rollback() # Rollback on error
             flash("Database error updating password.", "danger")
        # --- ADDED finally ---
        finally:
             if conn: conn.close() # Close connection

    # Render template for GET or failed POST validation
    return render_template("update_password.html", form=form)

@app.route('/delete_user_account', methods=['GET', 'POST'])
def delete_user_account():
    if 'user_id' not in session:
        flash("You need to be logged in to delete your account.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_id = session.get('user_id')
        full_name = session.get('full_name')
        if not user_id:
             flash("Session error, please log in again.", "warning")
             return redirect(url_for('login'))

        conn = get_db_connection()
        if not conn:
             flash("Database error, cannot delete account.", "danger")
             return render_template('delete_user.html')

        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
                conn.commit() # Commit deletion
            logging.warning(f"User account deleted: ID={user_id}, Full Name={full_name}")

            # --- Delete embedding file logic ---
            embedding_path = f"D:/Continuous_auth_system/user_embeddings/{user_id}_face.npy"
            if os.path.exists(embedding_path):
                try:
                    os.remove(embedding_path)
                    logging.info(f"Deleted embedding file for user {user_id}")
                except OSError as e:
                    logging.error(f"Error deleting embedding file {embedding_path}: {e}")

            # --- Clear session and global state ---
            session.clear()
            with auth_status_lock:
                if auth_status.get('user_id') == user_id:
                    auth_status['authenticated'] = False
                    auth_status['user_id'] = None

            flash("Your account has been permanently deleted.", "success")
            return redirect(url_for('home'))
        # --- ADDED except ---
        except pymysql.MySQLError as e:
            logging.error(f"Database error deleting account for user {user_id}: {e}")
            if conn: conn.rollback() # Rollback on error
            flash("Could not delete account due to a database error.", "danger")
        # --- ADDED finally ---
        finally:
            if conn: conn.close() # Close connection

    # For GET request, show confirmation page
    return render_template('delete_user.html')

@app.route("/user_logs")
def user_logs():
    if "user_id" not in session:
        flash("Please login to view logs.", "info")
        return redirect(url_for('login'))

    user_id = session["user_id"]
    logs = []
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT timestamp, event_type, status, details FROM authlogs WHERE user_id = %s ORDER BY timestamp DESC LIMIT 100",
                    (user_id,)
                )
                logs = cursor.fetchall()
        except pymysql.MySQLError as e:
            logging.error(f"DB error fetching logs for user {user_id}: {e}")
            flash("Could not load activity logs.", "warning")
        # --- ADDED finally ---
        finally:
            if conn: conn.close() # Close connection

    return render_template("user_logs.html", logs=logs)

@app.route("/analytics")
def analytics_dashboard():
    """Render the analytics dashboard."""
    if 'user_id' not in session:
        flash("Please login to view analytics.", "warning")
        return redirect(url_for('login'))
    return render_template("analytics_dashboard.html")

def parse_date_params():
    """Helper to parse and validate date parameters."""
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=6)

    start_str = request.args.get('startDate', default_start.strftime('%Y-%m-%d'))
    end_str = request.args.get('endDate', default_end.strftime('%Y-%m-%d'))

    try:
        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date_inclusive = datetime.strptime(end_str, '%Y-%m-%d').date() + timedelta(days=1)
        if end_date_inclusive <= start_date:
            end_date_inclusive = start_date + timedelta(days=1)
        return start_date, end_date_inclusive
    except ValueError:
        logging.warning(f"Invalid date format: start='{start_str}', end='{end_str}'. Using defaults.")
        return default_start, default_end + timedelta(days=1)

@app.route("/api/analytics/logins_over_time")
def api_logins_over_time():
    if 'user_id' not in session:
        abort(401)

    start_date, end_date_inclusive = parse_date_params()
    conn = get_db_connection()
    data = {'labels': [], 'success_data': [], 'fail_data': []}
    if not conn:
        return jsonify(data), 500

    try:
        with conn.cursor() as cursor:
            query = """
                SELECT DATE(timestamp) as log_date, status, COUNT(*) as count
                FROM authlogs
                WHERE event_type IN ('login_success', 'login_fail')
                  AND timestamp >= %s AND timestamp < %s
                GROUP BY log_date, status
                ORDER BY log_date ASC;
            """
            cursor.execute(query, (start_date, end_date_inclusive))
            results = cursor.fetchall()

            processed_data = {}
            for row in results:
                date_str = row['log_date'].strftime('%Y-%m-%d')
                if date_str not in processed_data:
                    processed_data[date_str] = {'success': 0, 'fail': 0}
                if date_str not in processed_data:
                    processed_data[date_str] = {'success': 0, 'fail': 0}
                if row['status'] == 'success':
                    processed_data[date_str]['success'] = row['count']
                elif row['status'] == 'fail':
                    processed_data[date_str]['fail'] = row['count']

            current_date = start_date
            while current_date < end_date_inclusive:
                date_str = current_date.strftime('%Y-%m-%d')
                counts = processed_data.get(date_str, {'success': 0, 'fail': 0})
                data['labels'].append(date_str)
                data['success_data'].append(counts['success'])
                data['fail_data'].append(counts['fail'])
                current_date += timedelta(days=1)
    except pymysql.MySQLError as e:
        logging.error(f"DB Error fetching login trends: {e}")
        return jsonify({"error": "Database query failed"}), 500
    finally:
        if conn:
            conn.close()

    return jsonify(data)

@app.route("/api/analytics/success_rate")
def api_success_rate():
    if 'user_id' not in session:
        abort(401)

    start_date, end_date_inclusive = parse_date_params()
    conn = get_db_connection()
    data = {'success': 0, 'fail': 0}
    if not conn:
        return jsonify(data), 500

    try:
        with conn.cursor() as cursor:
            query = """
                SELECT status, COUNT(*) as count
                FROM authlogs
                WHERE event_type IN ('login_success', 'login_fail')
                  AND timestamp >= %s AND timestamp < %s
                GROUP BY status;
            """
            cursor.execute(query, (start_date, end_date_inclusive))
            results = cursor.fetchall()

            for row in results:
                if row['status'] == 'success':
                    data['success'] = row['count']
                elif row['status'] == 'fail':
                    data['fail'] = row['count']
    except pymysql.MySQLError as e:
        logging.error(f"DB Error fetching success rate: {e}")
        return jsonify({"error": "Database query failed"}), 500
    finally:
        if conn:
            conn.close()

    return jsonify(data)

# In app.py

# In app.py

@app.route("/api/analytics/logins_per_user")
@login_required # Or @admin_required if only admins see analytics
def api_logins_per_user():
    start_date, end_date_exclusive = parse_date_params()
    conn = get_db_connection()
    data = []

    if not conn:
        logging.error("API Logins Per User: Database connection failed.")
        return jsonify(error="Database connection failed"), 500

    try:
        with conn.cursor() as cursor:
            # Ensure this query is exactly as follows:
            query = """
                SELECT u.username, COUNT(a.log_id) as login_count
                FROM authlogs a
                JOIN users u ON a.user_id = u.user_id
                WHERE a.event_type = 'login_success' AND a.status = 'success'
                  AND a.timestamp >= %s AND a.timestamp < %s
                GROUP BY u.user_id, u.username  # Group by user_id and username
                ORDER BY login_count DESC
                LIMIT 20;
            """
            # Log the query and params for debugging if needed
            # logging.debug(f"Executing query for logins_per_user: {query} with params: ({start_date}, {end_date_exclusive})")
            
            cursor.execute(query, (start_date, end_date_exclusive))
            data = cursor.fetchall()
            # logging.debug(f"Data fetched for logins_per_user: {data}")

    except pymysql.MySQLError as e:
        logging.error(f"API Logins Per User: Database query error: {e}", exc_info=True)
        return jsonify({"error": "Database query failed"}), 500
    except Exception as e:
        # Catch any other unexpected errors during processing
        logging.error(f"API Logins Per User: Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred"}), 500
    finally:
        if conn:
            conn.close()

    return jsonify(data)

@app.route("/test_db_connection")
@limiter.limit("5 per hour")
def test_db_connection_route():
    """Endpoint to explicitly test DB connection."""
    if is_mysql_server_reachable(MYSQL_HOST):
        return jsonify({"status": "success", "message": "Database connection successful!"}), 200
    else:
        return jsonify({"status": "error", "message": "Database connection failed."}), 500

@app.route("/test")
def test_route():
    """Simple route for testing server availability."""

@app.route('/static/<path:filename>')
def serve_static(filename):
     """Explicitly define route for static files (Flask does this by default)."""
     return send_from_directory('static', filename)

@app.route("/sidebar")
def sidebar():
    """Serves the sidebar HTML."""
    return render_template("sidebar.html")

@app.route('/toggle_theme', methods=['POST'])
def toggle_theme():
    """Toggle between light and dark mode."""
    current_theme = session.get('theme', 'light')
    new_theme = 'dark' if current_theme == 'light' else 'light'
    session['theme'] = new_theme
    return jsonify({'theme': new_theme})

# --- Error Handlers ---

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message="Page Not Found", error_code=404), 404

@app.errorhandler(500)
def internal_error(error):
    # Log the error details
    logging.error(f"Internal Server Error: {error}", exc_info=True)
    # Don't rollback here unless you know the state
    return render_template('error.html', message="Internal Server Error", error_code=500), 500

@app.errorhandler(400)
def bad_request_error(error):
    # Check if it's a CSRF error
    if "CSRF" in str(error.description):
         logging.warning(f"CSRF Error: {error.description}")
         return render_template('error.html', message=f"Security token error: {error.description}. Please try again.", error_code=400), 400
    logging.error(f"Bad Request Error: {error}", exc_info=True)
    return render_template('error.html', message=f"Bad Request: {error.description}", error_code=400), 400

@app.errorhandler(401)
def unauthorized_error(error):
     flash("You need to be logged in to access this page.", "warning")
     return redirect(url_for('login', next=request.url))

@app.errorhandler(403)
def forbidden_error(error):
     return render_template('error.html', message="Forbidden - You don't have permission to access this.", error_code=403), 403

# --- NEW Route: Main Admin Dashboard Page ---

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'admin':
            flash("You do not have permission to access this page.", "danger")
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Renders the main Admin Dashboard page."""
    return render_template("admin_dashboard.html")

# --- NEW API Endpoints for Admin ---

@app.route("/api/admin/stats")
@admin_required
def api_admin_stats():
    """API endpoint to fetch system stats."""
    conn = get_db_connection()
    stats = {'total_users': 0, 'total_logins': 0, 'total_failures': 0}
    if not conn:
        return jsonify(stats), 500

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as total_users FROM users")
            stats['total_users'] = cursor.fetchone()['total_users']

            cursor.execute("SELECT COUNT(*) as total_logins FROM authlogs WHERE event_type = 'login_success'")
            stats['total_logins'] = cursor.fetchone()['total_logins']

            cursor.execute("SELECT COUNT(*) as total_failures FROM authlogs WHERE event_type = 'login_fail'")
            stats['total_failures'] = cursor.fetchone()['total_failures']
    except pymysql.MySQLError as e:
        logging.error(f"DB Error fetching admin stats: {e}")
        return jsonify({"error": "Database query failed"}), 500
    finally:
        if conn:
            conn.close()

    return jsonify(stats)

@app.route("/api/admin/users")
@admin_required
def api_admin_users():
    """API endpoint to fetch user list."""
    conn = get_db_connection()
    users_data = []
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn.cursor() as cursor:
            # Updated query to select full_name instead of username
            cursor.execute("SELECT user_id, full_name, email, role FROM users ORDER BY user_id ASC")
            users_data = cursor.fetchall()
    except pymysql.MySQLError as e:
        logging.error(f"DB Error fetching user list for admin: {e}")
        return jsonify({"error": "Database query failed"}), 500
    finally:
        if conn:
            conn.close()

    return jsonify(users_data)

@app.route("/api/admin/logs")
@admin_required
def api_admin_logs():
    """API endpoint to fetch system logs with filtering."""
    conn = get_db_connection()
    logs_data = []
    if not conn:
        return jsonify(logs_data), 500

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT log_id, user_id, event_type, status, details, timestamp FROM authlogs ORDER BY timestamp DESC LIMIT 100")
            logs_data = cursor.fetchall()
    except pymysql.MySQLError as e:
        logging.error(f"DB Error fetching system logs: {e}")
        return jsonify({"error": "Database query failed"}), 500
    finally:
        if conn:
            conn.close()

    return jsonify(logs_data)

# --- NEW Imports for Parallel Processing ---

import queue
from concurrent.futures import ThreadPoolExecutor

# --- NEW: Parallel Processing Variables ---

MAX_WORKER_THREADS = 3
MAX_QUEUE_SIZE = 5
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
results_queue = queue.Queue()
executor = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS + 1, thread_name_prefix='FaceWorker')
stop_event = threading.Event()
loaded_embeddings_cache = {}
embedding_cache_lock = threading.Lock()

# --- NEW: Function to Load Reference Embedding ---

def load_reference_embedding(user_id):
    if user_id is None:
        return None
    with embedding_cache_lock:
        if user_id in loaded_embeddings_cache:
            logging.debug(f"Using cached embedding for user {user_id}")
            return loaded_embeddings_cache[user_id]
    stored_encoding_path = f"D:/Continuous_auth_system/user_embeddings/{user_id}_face.npy"
    try:
        if os.path.exists(stored_encoding_path):
            embedding = np.load(stored_encoding_path)
            with embedding_cache_lock:
                loaded_embeddings_cache[user_id] = embedding
            return embedding
        else:
            logging.warning(f"Embedding file not found for user {user_id}")
    except Exception as e:
        logging.error(f"Error loading embedding for user {user_id}: {e}")
    return None

# --- NEW: Parallel Processing Worker Functions ---

def capture_frames_worker(cam_index=0):
    global cap
    logging.info("📷 Frame capture worker started.")
    target_fps = 5  # Reduced FPS to 5 for lower workload
    target_delay = 1.0 / target_fps
    frame_skip_interval = 2  # Capture every 2nd frame
    frame_skip_count = 0  # Counter to skip frames

    while not stop_event.is_set():
        frame = None
        with camera_lock:
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.warning("📷 Failed to read frame.")
                    frame = None
                    continue
            else:
                logging.warning("📷 Camera not ready. Attempting init...")
                if not initialize_camera(cam_index):
                    time.sleep(1)
                continue

        # Skip frames to reduce processing load
        frame_skip_count += 1
        if frame_skip_count % frame_skip_interval != 0:
            continue

        if frame is not None:
            try:
                frame_queue.put_nowait((frame.copy(), time.time()))
            except queue.Full:
                logging.debug("Frame queue full, dropping frame.")
        time.sleep(target_delay)
    logging.info("📷 Frame capture worker stopped.")
    release_camera()

def process_face_worker(worker_id):
    logging.info(f"👷 Face processing worker {worker_id} started.")
    while not stop_event.is_set():
        try:
            frame, timestamp = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        faces = detect_faces_mediapipe(frame)
        if faces:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            face_image_pil = extract_face_pil(face_roi)
            if face_image_pil is not None:
                current_embedding = get_face_embedding(face_image_pil)
                if current_embedding is not None:
                    with auth_status_lock:
                        user_id = auth_status.get('user_id')
                    if user_id is not None:
                        reference_embedding = load_reference_embedding(user_id)
                        if reference_embedding is not None:
                            match = verify_face(current_embedding, reference_embedding)
                            with auth_status_lock:
                                auth_status['authenticated'] = match
                                auth_status['last_check'] = time.time()
                                if not match: # If verification failed for this user
                                    auth_status['user_id'] = None # Add this line
    logging.info(f"👷 Face processing worker {worker_id} stopped.")

def start_face_processing_threads():
    if not initialize_camera():
        logging.error("Cannot start processing threads, camera failed to initialize.")
        return False
    stop_event.clear()
    executor.submit(capture_frames_worker)
    for i in range(MAX_WORKER_THREADS):
        executor.submit(process_face_worker, i + 1)
    logging.info("Background threads started.")
    return True

def stop_face_processing_threads():
    if stop_event.is_set():
        return
    stop_event.set()
    executor.shutdown(wait=True)
    while not frame_queue.empty():
        frame_queue.get_nowait()
    while not results_queue.empty():
        results_queue.get_nowait()
    logging.info("Background threads stopped.")

@app.route('/some_form_route', methods=['GET', 'POST'])
def some_form_route():
    from flask_wtf import FlaskForm
    from wtforms import StringField, SubmitField
    from wtforms.validators import DataRequired
    from wtforms import StringField, SubmitField
    class SomeForm(FlaskForm):
        example_field = StringField("Example Field", validators=[DataRequired()])
        submit = SubmitField("Submit")
        example_field = StringField("Example Field", validators=[DataRequired()])
    form = SomeForm()
    if form.validate_on_submit():
        # ...existing code...
        pass
    return render_template('some_form.html', form=form)

@app.route('/some_ajax_route', methods=['POST'])
def some_ajax_route():
    # Ensure CSRF token is validated
    # ...existing code...
    pass

from flask import redirect, url_for, session, flash

@app.route('/get_monitored_urls')
@login_required
def get_monitored_urls():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        logging.error(f"DB connection failed for get_monitored_urls user {user_id}")
        return jsonify([]), 500  # Return empty list on error
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT monitor_id, url, status, alert_count, last_checked
                FROM monitored_urls
                WHERE user_id = %s
                ORDER BY created_at DESC
            """
            cursor.execute(sql, (user_id,))
            sites = cursor.fetchall()
            # Convert datetime objects to strings for JSON serialization
            for site in sites:
                if site.get('last_checked'):
                    site['last_checked'] = site['last_checked'].isoformat()
            return jsonify(sites), 200
    except pymysql.MySQLError as e:
        logging.error(f"DB error in get_monitored_urls for user {user_id}: {e}")
        return jsonify(error="Database query failed"), 500
    except Exception as e:
        logging.error(f"Unexpected error in get_monitored_urls: {e}", exc_info=True)
        return jsonify(error="An unexpected error occurred"), 500
    finally:
        if conn:
            conn.close()
# --- NEW Imports for APScheduler ---
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from requests.exceptions import RequestException
import atexit
# --- Database Helper Functions ---
def add_monitored_site_db(user_id, url):
    conn = get_db_connection()
    if not conn:
        return False, "Database connection error"
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO monitored_urls (user_id, url) VALUES (%s, %s)",  # Changed table name
                (user_id, url)
            )
        conn.commit()
        return True, "Site added successfully."
    except pymysql.MySQLError as e:
        conn.rollback()
        if e.args[0] == 1062:  # Duplicate entry
            return False, "You are already monitoring this URL."
        return False, "Database error."
    finally:
        conn.close()

def get_monitored_sites_db(user_id):
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT monitor_id, url, status, alert_count, last_checked, last_known_hash, created_at "  # Updated columns
                "FROM monitored_urls WHERE user_id = %s ORDER BY created_at DESC",  # Changed table name
                (user_id,)
            )
            return cursor.fetchall()
    except pymysql.MySQLError:
        return []
    finally:
        conn.close()

def remove_monitored_site_db(user_id, site_id):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM monitored_urls WHERE monitor_id = %s AND user_id = %s",  # Changed table name and column
                (site_id, user_id)
            )
        conn.commit()
        return cursor.rowcount > 0
    except pymysql.MySQLError:
        conn.rollback()
        return False
    finally:
        conn.close()

def update_site_status_db(site_id, status_code, error=None):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE monitored_urls SET last_checked = NOW(), status = %s, alert_count = alert_count + 1, last_known_hash = %s WHERE monitor_id = %s",  # Changed table name and columns
                (status_code, error, site_id)
            )
        conn.commit()
    except pymysql.MySQLError:
        conn.rollback()
    finally:
        conn.close()

def check_website(site_id, url):
    try:
        response = requests.get(url, timeout=10)
        update_site_status_db(site_id, response.status_code)
    except RequestException as e:
        update_site_status_db(site_id, None, str(e)[:255])

def perform_all_checks():
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("SELECT monitor_id, url FROM monitored_urls")  # Changed table name
            sites = cursor.fetchall()
        for site in sites:
            check_website(site['monitor_id'], site['url'])  # Updated column name
    finally:
        conn.close()
@app.route('/monitoring/manage', methods=['GET', 'POST'])
@login_required
def manage_monitoring_page():
    """Handles monitoring management logic."""
    user_id = session['user_id'] # Relies on @login_required to ensure user_id exists
    if request.method == 'POST':
        url_to_monitor = request.form.get('url', '').strip()
    if request.method == 'POST':
        # CSRF token is typically handled by Flask-WTF automatically if CSRFProtect(app) is enabled.
        # If the POST request is rejected due to CSRF, it would usually result in a 400 error.
        # CSRF token is typically handled by Flask-WTF automatically if CSRFProtect(app) is enabled.
        if not url_to_monitor:
            flash("URL cannot be empty.", "warning")
        elif not validators.url(url_to_monitor): # Use the imported 'validators'
            flash(f"Invalid URL format: '{url_to_monitor}'. Please include http:// or https://.", "warning")
        else:
            # Attempt to add the site to the databasetor}'. Please include http:// or https://.", "warning")
            success, message = add_monitored_site_db(user_id, url_to_monitor)
            flash(message, 'success' if success else 'danger')
        # Always redirect after a POST to prevent issues with form resubmission
        return redirect(url_for('manage_monitoring_page'))
    # For GET requests:
    monitored_sites = get_monitored_sites_db(user_id)
    return render_template('monitoring.html', monitored_sites=monitored_sites)

@app.route('/monitoring/remove/<int:site_id>', methods=['POST'])
@login_required
def remove_monitoring(site_id):
    user_id = session['user_id']
    if remove_monitored_site_db(user_id, site_id):
        flash("Website removed from monitoring.", 'success')
    else:
        flash("Failed to remove website.", 'danger')
    return redirect(url_for('manage_monitoring_page'))

# --- Scheduler Initialization ---
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(perform_all_checks, 'interval', minutes=5)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

@app.route("/enroll_face/<int:user_id>", methods=["GET"])
def enroll_face_page(user_id):
    """Serves the page for new users to enroll their face."""
    # Optional: Verify if this user_id actually needs enrollment or exists
    conn = get_db_connection()
    user_exists = False
    full_name = "New User"
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT full_name FROM users WHERE user_id = %s", (user_id,))
                user = cursor.fetchone()
                if user:
                    full_name = user['full_name']
                    user_exists = True
        except pymysql.MySQLError as e:
            logging.error(f"DB error checking user {user_id} for enrollment: {e}")
        finally:
            conn.close()
    if not user_exists:
        return redirect(url_for('register')) # Or home/login
    return render_template("enroll_face.html", user_id=user_id, full_name=full_name)

def generate_enroll_frames():
    """Generates frames for the enrollment video feed."""
    while True:
        frame = None
        with camera_lock:
            if cap and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    frame = None
            else:
                initialize_camera() # Try to reinit if not open

        if frame is None:
            # Create a black frame or a "camera unavailable" image
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Error", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            time.sleep(1)  # Avoid busy-looping if camera is dead
            continue

        # --- Enrollment specific instructions ---
        h_frame, w_frame, _ = frame.shape
        center_x, center_y = w_frame // 2, h_frame // 2
        rect_w, rect_h = w_frame // 3, h_frame // 2 # Adjust size as needed

        # Draw a rectangle or oval as a guide for face placement
        cv2.rectangle(frame, (center_x - rect_w//2, center_y - rect_h//2), (center_x + rect_w//2, center_y + rect_h//2), (0, 255, 0), 2)
        cv2.putText(frame, "Position face in rectangle and capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)  # Limit frame rate slightly

@app.route("/capture_enroll_face/<int:user_id>", methods=["POST"])
@csrf.exempt # Temporarily exempt if CSRF is an issue here, or ensure token is sent
def capture_enroll_face(user_id):
    """Captures the current frame, extracts face, and saves embedding for the given user_id."""
    frame = None
    with camera_lock:
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame = None
        else:
             initialize_camera() # Try to reinit if not open

    if frame is None:
        logging.warning(f"Failed to capture frame for face enrollment (User ID: {user_id})")
        return jsonify({'status': 'error', 'message': 'Camera not available.'}), 500

    # --- Face Detection and Embedding ---
    detected_faces = detect_faces_mediapipe(frame)
    if not detected_faces:
        logging.warning(f"No face detected in enrollment frame for user {user_id}.")
        return jsonify({'status': 'fail', 'message': 'No face detected. Please ensure your face is clearly visible.'}), 200 # 200 OK, but with a fail status

    x, y, w, h = detected_faces[0]
    face_roi = frame[y:y+h, x:x+w]
    face_image_pil = extract_face_pil(face_roi)
    if face_image_pil is None:
        logging.error(f"Could not convert detected face to PIL image for user {user_id}.")
        return jsonify({'status': 'error', 'message': 'Error processing face image.'}), 500

    embedding = get_face_embedding(face_image_pil)
    if embedding is None:
        logging.error(f"Could not generate face embedding for user {user_id} during enrollment.")
        return jsonify({'status': 'error', 'message': 'Could not generate face profile. Try a different pose or lighting.'}), 500

    # --- Save the embedding and respond ---
    save_face_encoding(user_id, embedding) # Your existing function
    logging.info(f"Face enrollment successful for user ID {user_id}.")
    # Clear the cache for this user if it exists, so it's reloaded next time
    with embedding_cache_lock:
        if user_id in loaded_embeddings_cache:
            logging.debug(f"Cleared cached embedding for user {user_id} after enrollment.")
            del loaded_embeddings_cache[user_id]
    return jsonify({'status': 'success', 'message': 'Face enrollment successful! Redirecting to login...'})