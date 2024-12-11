# Brainwave API

Brainwave API is a FastAPI-based backend service designed to handle authentication and other functionalities, with Firebase integration for data management.

## Features

- **FastAPI Framework**: A modern, fast (high-performance) web framework for building APIs with Python.
- **CORS Support**: Configured to allow requests from all origins for ease of integration.
- **Authentication**: Includes a modular authentication system.
- **Firebase Integration**: Uses Firebase for secure and scalable database management.
- **Environment Configuration**: Leveraging `.env` files for secure configuration management.

## Requirements

To run this project, you'll need:

- Python 3.7+
- A Firebase project with Secret Manager enabled
- A `.env` file with the necessary environment variables

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/brainwave-api.git
    cd brainwave-api
    ```

2. **Set up a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Add environment variables**:
    Create a `.env` file in the project root and add the required variables for Firebase initialization.

5. **Run the server**:
    ```bash
    uvicorn app:app --reload
    ```

## API Endpoints

### Home
- **GET** `/`
  - Returns a JSON message confirming the API is running.
  - Response:
    ```json
    {
      "message": "Brainwave API is up and running!"
    }
    ```

### Authentication
- **Base URL**: `/auth`
  - All authentication-related routes are prefixed with `/auth`.

## Middleware

The API uses the following middleware:
- **CORS Middleware**: Allows cross-origin requests from any source.


