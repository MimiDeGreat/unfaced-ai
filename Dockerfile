# Use Python 3.9 (required for TensorFlow 2.12)
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with COMPATIBLE versions
RUN pip install --upgrade pip &&
    pip install --no-cache-dir
    streamlit==1.28.0
    deepface==0.0.79
    tensorflow==2.12.1
    opencv-python-headless==4.7.0.72
    numpy==1.23.5
    soundfile==0.12.1
    librosa==0.10.0
    resemblyzer==0.1.3
    protobuf==3.20.3  # Required by TF 2.12

# Copy application files
COPY . .

# Apply critical DeepFace patch
RUN sed -i "s/return int(tf.__version__.split(\".\", maxsplit=1)[0]/return 2  # Force TF 2.x/" /usr/local/lib/python3.9/site-packages/deepface/commons/package_utils.py

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]