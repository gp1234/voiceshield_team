# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set environment variables
# Prevents Python from writing pyc files to disc (optional)
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to terminal without being buffered (optional)
ENV PYTHONUNBUFFERED 1
# Set a writable directory for Matplotlib's font cache
ENV MPLCONFIGDIR /tmp/matplotlib_cache

# 3. Install system dependencies
# - ffmpeg is crucial for pydub to handle various audio formats
# - build-essential might be needed for some Python packages that compile from source
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory in the container
WORKDIR /app

# 5. Copy the requirements file from your app directory first
# This leverages Docker's layer caching. If requirements.txt doesn't change,
# this layer won't be rebuilt, speeding up subsequent builds.
COPY ./app/requirements.txt /app/requirements.txt

# 6. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your application code from the app directory into the container
# This includes main.py, saved_models/, static_frontend/
COPY ./app/ /app/

# 8. Expose the port the app runs on
# This should match the port Uvicorn will listen on
EXPOSE 8000

# 9. Define the command to run your application
# This will be executed when the container starts.
# Uvicorn will serve your FastAPI app (main:app)
# --host 0.0.0.0 makes it accessible from outside the container
# --port 8000 matches the EXPOSE instruction
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
