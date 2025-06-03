import os

class Config:
    # Load environment variables
    SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_USER = os.getenv("MYSQL_USER", "flaskuser")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "Pravin0606")
    MYSQL_DB = os.getenv("MYSQL_DB", "continuous_auth")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Updated session configuration
    SESSION_COOKIE_NAME = "your_app_session"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_PATH = "/"
    SESSION_COOKIE_DOMAIN = None
    SESSION_COOKIE_MAX_AGE = 3600  # 1 hour expiration
    SESSION_USE_SIGNER = True
    SESSION_PERMANENT = False
    SESSION_TYPE = "filesystem"  # Or "sqlalchemy" or "redis" if using database
