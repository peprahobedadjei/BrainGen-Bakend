import uuid
import re
from werkzeug.security import generate_password_hash

def validate_email(email):
    regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.fullmatch(regex, email)

def validate_password(password):
    return len(password) >= 8 and any(c.isupper() for c in password) and any(c.islower() for c in password) and any(c.isdigit() for c in password)

def hash_password(password):
    return generate_password_hash(password)

def create_user_pseudo_id():
    return str(uuid.uuid4())
