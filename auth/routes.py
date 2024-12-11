import random
import requests
from fastapi import APIRouter, HTTPException, Request
from firebase_admin import firestore
from google.cloud.firestore_v1 import FieldFilter
from core.utils import validate_email, validate_password, hash_password, create_user_pseudo_id
from database.firebase import db
import logging
from werkzeug.security import check_password_hash
# Set up logging
logger = logging.getLogger(__name__)

auth_bp = APIRouter()

# EmailJS Configuration
EMAILJS_SERVICE_ID = "service_x99amar"  
EMAILJS_TEMPLATE_ID = "template_z5u491g"  
EMAILJS_PRIVATE_KEY = "IJrEFPYStsEEVDtWKrBH5"
EMAILJS_PUBLIC_KEY = "k8vT8VLnWmpPiMEbN"

def send_email_with_emailjs(to_name, otp, from_name, reply_to):
    """
    Sends an email using EmailJS.
    """
    url = "https://api.emailjs.com/api/v1.0/email/send"
    payload = {
        "service_id": EMAILJS_SERVICE_ID,
        "template_id": EMAILJS_TEMPLATE_ID,
        "user_id": EMAILJS_PUBLIC_KEY,
        "accessToken": EMAILJS_PRIVATE_KEY,
        "template_params": {
            "to_name": to_name,
            "otp": otp,
            "from_name": from_name,
            "reply_to": reply_to,
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
        logger.info(f"Verification email sent successfully to {reply_to}")
    except requests.RequestException as e:
        logger.error(f"Failed to send email to {reply_to}: {e}")
        raise Exception(f"Failed to send email: {e}")

@auth_bp.post("/signup")
async def signup(request: Request):
    data = await request.json()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    confirm_password = data.get("confirm_password", "").strip()

    # Validate inputs
    if not username or not email or not password:
        logger.warning("Signup attempt with missing fields.")
        raise HTTPException(status_code=400, detail="All fields are required.")
    if not validate_email(email):
        logger.warning(f"Invalid email format: {email}")
        raise HTTPException(status_code=400, detail="Invalid email format.")
    if password != confirm_password:
        logger.warning(f"Passwords do not match for email: {email}")
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # Check if user exists
    users_ref = db.collection("users")
    existing_users = users_ref.where(filter=FieldFilter("email", "==", email)).get()
    
    if existing_users:
        logger.warning(f"Signup attempt with already registered email: {email}")
        raise HTTPException(status_code=400, detail="Email is already registered.")

    # Generate a 6-digit verification code
    verification_code = random.randint(100000, 999999)

    # Save user data
    user_pseudo_id = create_user_pseudo_id()
    hashed_password = hash_password(password)
    user_data = {
        "user_pseudo_id": user_pseudo_id,
        "username": username,
        "email": email,
        "password": hashed_password,
        "verification_code": verification_code,
        "is_verified": False,  # Set to False by default
        "created_at": firestore.SERVER_TIMESTAMP,
    }

    # Add a document with a custom document ID (email)
    try:
        users_ref.document(email).set(user_data)
        logger.info(f"User data saved successfully for email: {email}")
    except Exception as e:
        logger.error(f"Failed to save user data for {email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save user data. Please try again later.")

    # Send the verification email
    try:
        send_email_with_emailjs(username, verification_code, 'BrainGen', email)
    except Exception as e:
        # Handle email sending failure
        logger.error(f"Failed to send verification email to {email}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send verification email: {str(e)}")

    return {
        "message": "User registered successfully. Please verify your email with the verification code.",
        "user_pseudo_id": user_pseudo_id,
    }


@auth_bp.post("/activate")
async def activate(request: Request):
    data = await request.json()
    email = data.get("email", "").strip()
    verification_code = data.get("verification_code", "").strip()

    # Validate inputs
    if not email or not verification_code:
        logger.warning("Activation attempt with missing fields.")
        raise HTTPException(status_code=400, detail="Email and verification code are required.")

    # Fetch user data from Firestore
    users_ref = db.collection("users")
    user_doc = users_ref.document(email).get()

    if not user_doc.exists:
        logger.warning(f"Activation attempt for non-existent email: {email}")
        raise HTTPException(status_code=404, detail="User with email not found not found.")

    user_data = user_doc.to_dict()

    if str(user_data.get("verification_code")) != None:
        logger.warning(f"Account already activated: {email}")
        raise HTTPException(status_code=400, detail="User Account has be already activated")

    # Check if the verification code matches
    if str(user_data.get("verification_code")) != verification_code:
        logger.warning(f"Invalid verification code for email: {email}")
        raise HTTPException(status_code=400, detail="Invalid verification code.")


    # Update is_verified field
    try:
        users_ref.document(email).update({
            "is_verified": True,
            "verification_code": None  # Remove the verification code after successful activation
        })
        logger.info(f"User successfully verified: {email}")
    except Exception as e:
        logger.error(f"Failed to update verification status for {email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update verification status. Please try again later.")

    return {"message": "Account successfully verified. Get started with BarinGen Today"}



@auth_bp.post("/login")
async def login(request: Request):
    data = await request.json()
    username_or_email = data.get("username_or_email", "").strip()
    password = data.get("password", "").strip()

    # Validate inputs
    if not username_or_email or not password:
        logger.warning("Login attempt with missing fields.")
        raise HTTPException(status_code=400, detail="Account Username/Email and password are required.")

    # Fetch user data from Firestore
    users_ref = db.collection("users")
    query = users_ref.where(
        filter=FieldFilter("email", "==", username_or_email)
    ).get() or users_ref.where(
        filter=FieldFilter("username", "==", username_or_email)
    ).get()

    if not query:
        logger.warning(f"Login attempt with non-existent username/email: {username_or_email}")
        raise HTTPException(status_code=404, detail="User Account does not exit")

    user_doc = query[0]  # Assuming username or email is unique
    user_data = user_doc.to_dict()

    # Check if the account is verified
    if not user_data.get("is_verified") or user_data.get("verification_code") is not None:
        logger.warning(f"Login attempt with unverified account: {username_or_email}")
        raise HTTPException(status_code=403, detail="Account not verified. Please verify your account to log in.")

    # Verify the password
    if not check_password_hash(user_data.get("password"), password):
        logger.warning(f"Incorrect password attempt for: {username_or_email}")
        raise HTTPException(status_code=401, detail="User Account Password Incorrect.")

    # Successful login
    logger.info(f"User logged in successfully: {username_or_email}")
    return {
        "message": "Login successful.",
        "data": {
            "username": user_data["username"],
            "email": user_data["email"],
            "user_pseudo_id": user_data["user_pseudo_id"],
            "created_at": user_data["created_at"],
        },
    }