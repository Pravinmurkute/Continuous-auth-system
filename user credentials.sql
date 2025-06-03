-- Create the database
CREATE DATABASE IF NOT EXISTS continuous_auth;
USE  continuous_auth;

-- User Table
CREATE TABLE Users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face Data Table
CREATE TABLE FaceData (
    face_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    face_embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE ON UPDATE CASCADE
);

-- Voice Data Table
CREATE TABLE VoiceData (
    voice_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    voice_embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE ON UPDATE CASCADE
);

-- Authentication Logs Table
drop table AuthLogs;
CREATE TABLE AuthLogs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    event_type VARCHAR(50) NOT NULL,          -- More flexible than ENUM
    status VARCHAR(20) NOT NULL,              -- Flexible status field
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details VARCHAR(500),                     -- Optimized for short text details
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE ON UPDATE CASCADE
);

-- Add index on timestamp for better query performance



CREATE INDEX idx_timestamp ON AuthLogs(timestamp);

CREATE TABLE monitored_urls (
    monitor_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    url VARCHAR(1024) NOT NULL, -- Increased length for potentially long URLs
    status VARCHAR(50) NOT NULL DEFAULT 'active', -- e.g., 'active', 'paused', 'alerted'
    alert_count INT NOT NULL DEFAULT 0,
    last_checked DATETIME NULL, -- Will be updated by the worker later
    last_known_hash VARCHAR(64) NULL, -- For SHA-256 hash, will be used later
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE, -- Link to users table
    UNIQUE KEY unique_user_url (user_id, url(255)) -- Prevent a user monitoring the same URL twice (index limited length)
);

show tables;

ALTER TABLE users
ADD CONSTRAINT chk_username_format
CHECK (username REGEXP '^[a-zA-Z0-9]+$');

ALTER TABLE AuthLogs
MODIFY event_type VARCHAR(50) NOT NULL DEFAULT 'login';

ALTER TABLE Users
ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user';

DESCRIBE users;
DESCRIBE authlogs;


select * from users;





