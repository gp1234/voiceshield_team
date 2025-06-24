# Use the official Coqui TTS CPU image
FROM ghcr.io/idiap/coqui-tts-cpu:latest

# Set working directory
WORKDIR /app

# Copy requirements.txt for additional dependencies
COPY requirements.txt .

# Install dependencies (optimized for ARM64)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script and transcripts.csv
COPY generate_xtts_audio.py .
COPY transcripts.csv .

# Set environment variable to accept Coqui license non-interactively
ENV COQUI_LICENSE=accept

# Run the script
CMD ["python", "generate_xtts_audio.py"]