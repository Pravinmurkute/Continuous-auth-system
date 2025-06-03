from flask_mail import Mail, Message
import os
import logging

def configure_mail(app):
    app.config["MAIL_SERVER"] = "smtp.gmail.com"
    app.config["MAIL_PORT"] = 587
    app.config["MAIL_USE_TLS"] = True
    app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
    app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
    return Mail(app)

def send_otp_email(mail, recipient_email, otp):
    try:
        msg = Message("Your OTP Code", sender=os.getenv("MAIL_USERNAME"), recipients=[recipient_email])
        msg.body = f"Your OTP code is {otp}"
        mail.send(msg)
        logging.info("OTP sent to email: %s", recipient_email)
    except Exception as e:
        logging.error("Failed to send OTP: %s", str(e))
        if "535" in str(e):
            logging.error("Invalid email credentials. Please check your MAIL_USERNAME and MAIL_PASSWORD.")
            logging.error("Ensure that 'Allow less secure apps' is enabled in your Google account settings.")
        raise
