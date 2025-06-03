# ...existing code...
from flask import session, flash, redirect, url_for

def check_user_session():
    if not session.get('user_id'):
        session.clear()  # Clear any corrupted session data
        flash('User not logged in', 'error')
        return redirect(url_for('login'))
# ...existing code...
