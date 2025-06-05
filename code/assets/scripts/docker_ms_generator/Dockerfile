# Use Python 3.10 slim image (ARM64-native for M1 Pro)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set unbuffered output for real-time logging
ENV PYTHONUNBUFFERED=1

# Copy requirements.txt
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Python script and transcripts.csv
COPY generate_valle_audio.py .
COPY transcripts.csv .

# Run the script
CMD ["python", "generate_valle_audio.py"]