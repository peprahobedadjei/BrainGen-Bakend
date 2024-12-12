from fastapi import APIRouter, HTTPException,Form, Request,UploadFile, File
import logging
from google.cloud import storage
import torch
from torchvision import models
from pydantic import BaseModel
import os
import tempfile
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List
from starlette.responses import JSONResponse
from io import BytesIO

# Set up logging
logger = logging.getLogger(__name__)

## Define FastAPI router
predict_bp = APIRouter()

# Global variable to store the loaded model
loaded_model = None

# Allowed MIME types for image files
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}

BUCKET_NAME = "the-challenge-433814.firebasestorage.app"
USERS_DB_FOLDER = "users_db"


# Input model for JSON parsing
class ModelInput(BaseModel):
    model_name: str

# Define the request schema
class GradCAMRequest(BaseModel):
    image_paths: List[str]
    targets: List[int]



@predict_bp.post("/{user_id}/create-folder", tags=["Users"])
def create_user_folder(user_id: str):
    """
    Create a folder for the user in the `users_db` directory.
    """
    try:
        # Initialize the storage client
        storage_client = storage.Client()

        # Specify the bucket name
        bucket_name = "the-challenge-433814.firebasestorage.app"
        user_db_folder = "users_db"
        logger.info(f"Accessing bucket: {bucket_name}")

        # Specify the user's folder path
        user_folder_path = f"{user_db_folder}/{user_id}/"

        # Access the bucket
        bucket = storage_client.get_bucket(bucket_name)

        # Check if the folder already exists
        existing_blob = bucket.get_blob(user_folder_path)
        if existing_blob:
            return {
                "status": "error",
                "message": f"Folder with user ID '{user_id}' already exists.",
            }

        # Create an empty blob to represent the folder
        blob = bucket.blob(user_folder_path)
        blob.upload_from_string("")  # Creates an empty "folder"

        return {"status": "success", "message": f"Folder created for user '{user_id}'."}
    except Exception as e:
        logger.error(f"Error creating user folder: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating user folder: {e}")

    


@predict_bp.post("/{user_id}/upload-images", tags=["Users"])
async def upload_to_drive(
    user_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload multiple files to the user's folder in Google Cloud Storage.
    """
    if not files or len(files) == 0:
        return JSONResponse(
            {"error": "No files were uploaded."},
            status_code=400,
        )

    try:
        # Access Google Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)

        # Check if the user's folder exists
        user_folder_path = f"{USERS_DB_FOLDER}/{user_id}/"
        blobs = list(bucket.list_blobs(prefix=user_folder_path))
        if not blobs:
            return JSONResponse(
                {"error": f"No folder has been created for user '{user_id}'."},
                status_code=400,
            )

        uploaded_files = []

        for file in files:
            # Read file content into memory
            file_content = file.file.read()
            file_path = f"{USERS_DB_FOLDER}/{user_id}/{file.filename}"  # Define file path
            blob = bucket.blob(file_path)

            # Upload file to Google Cloud Storage
            blob.upload_from_string(file_content, content_type=file.content_type)
            logger.info(f"Uploaded file: {file_path}")

            uploaded_files.append(file.filename)

        return JSONResponse(
            {
                "success": True,
                "message": f"Uploaded {len(uploaded_files)} files for user '{user_id}'.",
                "uploaded_files": uploaded_files,
            },
            status_code=200,
        )

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return JSONResponse(
            {"success": False, "error": f"File upload failed: {e}"},
            status_code=500,
        )



@predict_bp.get("/models", tags=["Models"])
def list_predict_models():
    """
    List all items in the 'predict_models/' folder in the bucket
    'the-challenge-433814.firebasestorage.app'.
    """
    try:
        # Initialize the storage client
        storage_client = storage.Client()

        # Specify the bucket name
        bucket_name = "the-challenge-433814.firebasestorage.app"
        logger.info(f"Accessing bucket: {bucket_name}")

        # Get the bucket
        bucket = storage_client.get_bucket(bucket_name)

        # List all blobs with the prefix 'predict_models/'
        prefix = "predict_models/"
        blobs = bucket.list_blobs(prefix=prefix)
        blob_names = [blob.name for blob in blobs]

        if not blob_names:
            logger.info(f"No items found in folder '{prefix}' in bucket '{bucket_name}'.")
            return {"status": "success", "bucket": bucket_name, "items": [], "message": "The folder is empty."}

        logger.info(f"Found {len(blob_names)} items in folder '{prefix}' in bucket '{bucket_name}'.")
        return {"status": "success", "bucket": bucket_name, "items": blob_names}

    except Exception as e:
        logger.error(f"Error occurred while listing items in the folder: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing folder items: {str(e)}")



@predict_bp.post("/models/load", tags=["Models"])
def load_model(input: ModelInput):
    global loaded_model  # Declare as global to make it accessible in other functions
    try:
        storage_client = storage.Client()
        bucket_name = "the-challenge-433814.firebasestorage.app"
        bucket = storage_client.get_bucket(bucket_name)

        model_name = input.model_name
        blob = bucket.blob(model_name)

        if not blob.exists():
            raise HTTPException(status_code=404, detail="Model file not found.")

        # Use a platform-independent temporary directory
        temp_dir = tempfile.gettempdir()
        local_model_path = os.path.join(temp_dir, model_name.split('/')[-1])
        logger.info(f"Downloading model to temporary path: {local_model_path}")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

        # Download the model
        blob.download_to_filename(local_model_path)
        logger.info(f"Model '{model_name}' downloaded successfully.")

        # Load and prepare the model
        loaded_model = models.resnet50(weights=None)  # Explicitly specify no weights
        num_features = loaded_model.fc.in_features
        loaded_model.fc = torch.nn.Linear(num_features, 2)
        loaded_model.load_state_dict(torch.load(local_model_path, weights_only=True))
        loaded_model.eval()

        return {"status": "success", "message": f"Model '{model_name}' is loaded and ready for inference."}

    except Exception as e:
        logger.error(f"Error occurred while loading the model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")



@predict_bp.post("/models/{user_id}/gradcam", tags=["Models"])
async def apply_gradcam(user_id: str):
    global loaded_model  # Access the global variable

    # Check if model is loaded
    if loaded_model is None:
        raise HTTPException(status_code=400, detail="Model is not loaded. Load the model first.")

    try:
        # Google Cloud Storage bucket configuration
        storage_client = storage.Client()
        bucket_name = "the-challenge-433814.firebasestorage.app"
        bucket = storage_client.get_bucket(bucket_name)
        user_folder = f"users_db/{user_id}/"
        predictions_folder = f"{user_folder}predictions/"

        # Check for images in the user's folder
        blobs = bucket.list_blobs(prefix=user_folder)
        image_blobs = [blob for blob in blobs if blob.name.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not image_blobs:
            raise HTTPException(status_code=404, detail=f"No images found in folder '{user_folder}'.")

        # Create a new prediction folder
        prediction_id = f"prediction_{len(list(bucket.list_blobs(prefix=predictions_folder)))}"
        prediction_folder = f"{predictions_folder}{prediction_id}/"
        original_folder = f"{prediction_folder}original/"
        output_folder = f"{prediction_folder}output/"

        bucket.blob(original_folder).upload_from_string("")  # Create original folder
        bucket.blob(output_folder).upload_from_string("")  # Create output folder

        visualizations = []

        for image_blob in image_blobs:
            # Download the image to a temporary file
            temp_image_path = os.path.join(tempfile.gettempdir(), os.path.basename(image_blob.name))
            image_blob.download_to_filename(temp_image_path)

            # Load and preprocess the image
            image = Image.open(temp_image_path).convert('RGB')

            # Resize the original image to match Grad-CAM dimensions (224, 224)
            original_image_resized = image.resize((224, 224))

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image_tensor = transform(image).unsqueeze(0)

            # Generate Grad-CAM
            target_layer = [loaded_model.layer4[-1]]
            cam = GradCAM(model=loaded_model, target_layers=target_layer)
            target_obj = [ClassifierOutputTarget(1)]  # Default target class for now
            grayscale_cam = cam(input_tensor=image_tensor, targets=target_obj)
            grayscale_cam = grayscale_cam[0, :]  # Get the first (and only) CAM result

            # Convert resized original image to NumPy array
            original_image_np = np.array(original_image_resized).astype(np.float32) / 255.0

            # Overlay Grad-CAM on the resized original image
            visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)

            # Save original and Grad-CAM output to the prediction folder
            original_image_name = os.path.basename(image_blob.name)
            gradcam_image_name = f"gradcam_{original_image_name}"

            # Save original image to Google Cloud Storage
            original_blob = bucket.blob(f"{original_folder}{original_image_name}")
            original_blob.upload_from_filename(temp_image_path)

            # Save Grad-CAM output to Google Cloud Storage
            temp_output_path = os.path.join(tempfile.gettempdir(), gradcam_image_name)
            Image.fromarray(visualization).save(temp_output_path)
            output_blob = bucket.blob(f"{output_folder}{gradcam_image_name}")
            output_blob.upload_from_filename(temp_output_path)

            visualizations.append({
                "original": f"https://storage.googleapis.com/the-challenge-433814.firebasestorage.app/{original_folder}{original_image_name}",
                "gradcam": f"https://storage.googleapis.com/the-challenge-433814.firebasestorage.app/{output_folder}{gradcam_image_name}"
            })

        return {"status": "success", "prediction_id": prediction_id, "visualizations": visualizations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Grad-CAM: {str(e)}")



@predict_bp.post("/models/clear-memory", tags=["Models"])
def clear_memory():
    """
    Clear the loaded model from memory.
    """
    global loaded_model
    try:
        if loaded_model is not None:
            # Reset the loaded model
            loaded_model = None
            logger.info("Loaded model cleared from memory.")
            return {"status": "success", "message": "Loaded model has been cleared from memory."}
        else:
            return {"status": "info", "message": "No model is currently loaded in memory."}
    except Exception as e:
        logger.error(f"Error occurred while clearing memory: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

@predict_bp.get("/{user_id}/list-folders", tags=["Users"])
def list_user_folders(user_id: str):
    """
    List all folders in the `users_db` directory that match the user's `user_id`.
    """
    try:
        # Initialize the storage client
        storage_client = storage.Client()

        # Specify the bucket name and user_db_folder
        bucket_name = "the-challenge-433814.firebasestorage.app"
        user_db_folder = "users_db"
        user_folder_path = f"{user_db_folder}/{user_id}/"

        logger.info(f"Accessing bucket: {bucket_name} and folder: {user_folder_path}")

        # Access the bucket
        bucket = storage_client.get_bucket(bucket_name)

        # List all blobs with the prefix `users_db/{user_id}/`
        blobs = bucket.list_blobs(prefix=user_folder_path)

        # Extract folder names by filtering for paths ending with "/"
        folders = {blob.name for blob in blobs if blob.name.endswith("/")}

        if not folders:
            return {
                "status": "success",
                "message": f"No folders found for user '{user_id}' in 'users_db'.",
                "folders": [],
            }

        logger.info(f"Found {len(folders)} folders for user '{user_id}' in 'users_db'.")
        return {
            "status": "success",
            "message": f"Folders listed successfully for user '{user_id}'.",
            "folders": list(folders),
        }
    except Exception as e:
        logger.error(f"Error listing folders for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Error listing folders: {e}")


@predict_bp.get("/{user_id}/list-files", tags=["Users"])
def list_user_files(user_id: str):
    """
    List all files in the user's folder in the `users_db` directory, excluding subfolders.
    """
    try:
        # Initialize the storage client
        storage_client = storage.Client()

        # Specify the bucket name and user folder path
        bucket_name = "the-challenge-433814.firebasestorage.app"
        user_db_folder = "users_db"
        user_folder_path = f"{user_db_folder}/{user_id}/"

        logger.info(f"Accessing bucket: {bucket_name} and folder: {user_folder_path}")

        # Access the bucket
        bucket = storage_client.get_bucket(bucket_name)

        # List all blobs in the user's folder
        blobs = bucket.list_blobs(prefix=user_folder_path)

        # Include only files directly in the user's folder (no subfolder files)
        files = [
            blob.name.split("/")[-1]
            for blob in blobs
            if not blob.name.endswith("/") and blob.name.count("/") == user_folder_path.count("/")
        ]

        if not files:
            return {
                "status": "success",
                "message": f"No files found in folder for user '{user_id}'.",
                "files": [],
            }

        logger.info(f"Found {len(files)} files in folder '{user_folder_path}' for user '{user_id}'.")
        return {
            "status": "success",
            "message": f"Files listed successfully for user '{user_id}'.",
            "files": files,
        }
    except Exception as e:
        logger.error(f"Error listing files for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {e}")

