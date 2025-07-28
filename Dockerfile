FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Install system dependencies for PyMuPDF and libcrypt fix
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmupdf-dev \
    libcrypt1 \
    && ln -s /lib/x86_64-linux-gnu/libcrypt.so.1 /lib/x86_64-linux-gnu/libcrypt.so.2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --timeout=100 --retries=10 --no-cache-dir -r requirements.txt

# Copy all Python files
COPY document_processor.py .
COPY section_extractor.py .
COPY persona_analyzer.py .
COPY main.py .

# Copy models directory
COPY models/ ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the main script
CMD ["python", "main.py"]
