# Use a lightweight Python image as the base
FROM python:3.10.12-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=True
ENV APP_HOME=/back-end

# Set the working directory
WORKDIR $APP_HOME

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for the application
EXPOSE 8080

# Command to run the FastAPI app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app", "-k", "uvicorn.workers.UvicornWorker"]
