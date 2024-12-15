from fastapi import APIRouter, HTTPException,Form, Request,UploadFile, File
import logging
from google.cloud import storage
from pydantic import BaseModel
import os
import tempfile
from typing import List
from starlette.responses import JSONResponse
from keras.models import load_model
from tensorflow.keras.layers import Conv2DTranspose
from PIL import Image
import numpy as np
import shutil

# Set up logging
logger = logging.getLogger(__name__)

## Define FastAPI router
generate_bp = APIRouter()

# Global variable to store the loaded model
loaded_generator_model = None
loaded_discriminator_model = None

BUCKET_NAME = "the-challenge-433814.firebasestorage.app"
USERS_DB_FOLDER = "users_db"

# Input model for JSON parsing
class ModelInput(BaseModel):
    model_name: str

# Define the input schema for the request body
class GenerateInput(BaseModel):
    num_images: int
    noise_dim: int

@generate_bp.post("/{user_id}/create-folder", tags=["Users"])
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
    

@generate_bp.post("/{user_id}/upload-images", tags=["Users"])
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


@generate_bp.get("/generate-models", tags=["Models"])
def list_predict_models():
    """
    List all items in the 'generator_models/' folder in the bucket
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

        # List all blobs with the prefix 'generator_models/'
        prefix = "generator_models/"
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
    

@generate_bp.get("/discriminator-models", tags=["Models"])
def list_predict_models():
    """
    List all items in the 'discriminator_models/' folder in the bucket
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

        # List all blobs with the prefix 'generator_models/'
        prefix = "discriminator_models/"
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
    

@generate_bp.post("/models/generate-load", tags=["Models"])
def load_generatemodel(input: ModelInput):
    global loaded_generator_model  # Declare as global to make it accessible in other functions
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

       # Handle Conv2DTranspose's unrecognized arguments
        def custom_conv2d_transpose(*args, **kwargs):
            kwargs.pop("groups", None)  # Safely remove unsupported parameters
            return Conv2DTranspose(*args, **kwargs)

        # Load the model with custom objects
        loaded_generator_model = load_model(local_model_path, custom_objects={
            'Conv2DTranspose': custom_conv2d_transpose
        })
        print(loaded_generator_model.summary())
        logger.info(f"Model '{model_name}' loaded and ready for inference.")
        return {"status": "success", "message": f"Model '{model_name}' is loaded and ready for inference."}

    except Exception as e:
        logger.error(f"Error occurred while loading the model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")
    

@generate_bp.post("/models/{user_id}/generate-images", tags=["Models"])
async def generate_images(user_id: str, input: GenerateInput):
    """
    Generate images using the loaded generator model and save them to Google Cloud Storage.
    Additionally, upload a local file (output.png) as a plot to the generated folder.
    """
    global loaded_generator_model  # Ensure the generator model is accessible

    if loaded_generator_model is None:
        raise HTTPException(status_code=400, detail="Generator model is not loaded. Please load the model first.")

    try:
        # Extract parameters from the request body
        num_images = input.num_images
        noise_dim = input.noise_dim

        # Initialize Google Cloud Storage
        storage_client = storage.Client()
        bucket_name = "the-challenge-433814.firebasestorage.app"
        bucket = storage_client.get_bucket(bucket_name)

        # Determine the folder for the current generation
        user_folder = f"users_db/{user_id}/"
        generation_folder_prefix = f"{user_folder}generative_"
        existing_generations = list(bucket.list_blobs(prefix=generation_folder_prefix))
        next_generation_id = len({blob.name.split('/')[2] for blob in existing_generations}) + 1
        generation_folder = f"{generation_folder_prefix}{next_generation_id}/"

        # Create an empty folder in GCS
        bucket.blob(generation_folder).upload_from_string("")

        # Generate noise and create images
        noise = np.random.normal(0, 1, size=(num_images, noise_dim))
        generated_images = loaded_generator_model.predict(noise)
        generated_images = (generated_images + 1) / 2.0  # Rescale to [0, 1]

        # Save and upload each image
        uploaded_images = []
        for i in range(num_images):
            temp_image_path = os.path.join(tempfile.gettempdir(), f"generated_{i + 1}.png")
            # Assuming grayscale images with a single channel
            image = (generated_images[i, :, :, 0] * 255).astype(np.uint8)
            Image.fromarray(image).save(temp_image_path)

            # Upload to GCS
            image_blob_name = f"{generation_folder}generated_{i + 1}.png"
            bucket.blob(image_blob_name).upload_from_filename(temp_image_path)

            # Append the public URL of the image
            uploaded_images.append({
                "image_name": f"generated_{i + 1}.png",
                "image_url": f"https://storage.googleapis.com/{bucket_name}/{image_blob_name}"
            })

        # Upload output.png as a plot
        plot_path = "./output.png"
        plot_url = None
        if os.path.exists(plot_path):
            plot_blob_name = f"{generation_folder}output.png"
            bucket.blob(plot_blob_name).upload_from_filename(plot_path)
            plot_url = f"https://storage.googleapis.com/{bucket_name}/{plot_blob_name}"

        return {
            "status": "success",
            "generation_id": next_generation_id,
            "num_images": num_images,
            "images": uploaded_images,
            "plot": plot_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


def clear_memory(temp_dir: str = None):
    """
    Clear temporary files and reset global variables used during image generation.

    Args:
        temp_dir (str): Path to the temporary directory to clear. If not provided, defaults to `tempfile.gettempdir()`.
    """
    global loaded_generator_model  # Access global model

    # Reset global variables
    loaded_generator_model = None

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    try:
        # Ensure the directory exists before attempting to delete
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    # Remove files
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    # Remove directories
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Could not delete {file_path}: {str(e)}")
        print(f"Cleared temporary files in directory: {temp_dir}")
        print("Global variables reset.")
        return {"status": "success", "message": f"Cleared temporary files in directory: {temp_dir}"}
    except Exception as e:
        print(f"Error clearing temporary files or resetting variables: {str(e)}")
        return {"status": "error", "message": f"Error clearing temporary files or resetting variables: {str(e)}"}
    

@generate_bp.post("/clear-memory", tags=["Utils"])
async def clear_memory_api(temp_dir: str = None):
    """
    Clear temporary files and reset global variables used during image generation.

    Args:
        temp_dir (str): Optional path to the temporary directory to clear.
    """
    result = clear_memory(temp_dir)
    if result["status"] == "success":
        return result
    else:
        raise HTTPException(status_code=500, detail=result["message"])
