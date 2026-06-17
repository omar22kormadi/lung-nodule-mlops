# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Set the working directory to /app
WORKDIR /app

# Install system dependencies required by opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*
    
# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start FastAPI server
# We run it from the root directory so the relative paths in api.py (e.g. parent.parent) work correctly.
CMD ["python", "-m", "uvicorn", "03_Model_Operations.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
