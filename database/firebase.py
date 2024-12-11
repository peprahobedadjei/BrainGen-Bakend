import json
from google.cloud import secretmanager
from firebase_admin import credentials, initialize_app, firestore, get_app, App

def get_secret(secret_id, project_id="539472932670"):
    """
    Fetch a secret value from Google Cloud Secret Manager.
    """
    try:
        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(name=secret_name)
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Error accessing Secret Manager: {e}")
        raise

def initialize_firebase():
    """
    Initialize Firebase using credentials from Secret Manager.
    """
    try:
        # Check if the default Firebase app is already initialized
        try:
            app = get_app()
            print("Firebase app already initialized.")
            return firestore.client()
        except ValueError:
            # App is not initialized yet
            pass

        # Fetch the secret containing the Firebase service account key
        firebase_key = get_secret("the-challenge-secret-key")

        # Parse the service account key JSON
        firebase_credentials = json.loads(firebase_key)

        # Initialize Firebase app
        cred = credentials.Certificate(firebase_credentials)
        initialize_app(cred)

        # Return the Firestore client
        return firestore.client()

    except Exception as e:
        print(f"Firebase initialization error: {e}")
        raise

# Initialize Firestore client
db = initialize_firebase()
