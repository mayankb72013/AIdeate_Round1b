FROM --platform=linux/amd64 python:3.9

WORKDIR /app

# Install system dependencies for PyMuPDF and other requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmupdf-dev \
    libcrypt1 \
    && ln -s /lib/x86_64-linux-gnu/libcrypt.so.1 /lib/x86_64-linux-gnu/libcrypt.so.2 \
    && rm -rf /var/lib/apt/lists/*

# Set offline environment variables to prevent network calls
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --timeout=100 --retries=10 --no-cache-dir -r requirements.txt

# Copy all Python files
COPY document_processor.py .
COPY section_extractor.py .
COPY persona_analyzer.py .
COPY main.py .

# Copy models directory (make sure this exists in your project)
COPY models/ ./models/

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Run the main script
CMD ["python", "main.py"]