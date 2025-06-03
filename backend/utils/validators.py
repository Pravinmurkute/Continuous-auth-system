import re

def is_valid_username(username):
    """Allow only characters, numbers, and a combination of both"""
    pattern = r'^[a-zA-Z0-9]+$'
    return bool(re.match(pattern, username))
