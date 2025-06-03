import os
import pymysql
import logging
from werkzeug.security import check_password_hash

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Database connection
def get_db_connection():
    try:
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB"),
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
    except pymysql.MySQLError as e:
        logging.error(f"Database connection error: {e}")
        return None

def verify_password(username, entered_password):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()

        if not user:
            logging.warning(f"User not found: {username}")
            return False

        stored_password_hash = user["password"]
        logging.debug(f"Stored password hash for user {username}: {stored_password_hash}")

        if check_password_hash(stored_password_hash, entered_password):
            logging.debug(f"Password for user {username} is correct")
            return True
        else:
            logging.warning(f"Invalid password for user: {username}")
            return False

    except Exception as e:
        logging.error(f"Error verifying password: {e}")
        return False

if __name__ == "__main__":
    username = input("Enter username: ")
    password = input("Enter password: ")
    if verify_password(username, password):
        print("Password is correct")
    else:
        print("Invalid password")
