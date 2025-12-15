FROM python:3.11-slim AS model_downloader
WORKDIR /tmp/model
# Install git-lfs if needed for large files
RUN apt-get update && apt-get install -y git-lfs && git lfs install
# Install huggingface-cli
RUN pip install huggingface-hub
# Use hf-transfer for faster downloads if desired (set HF_HUB_ENABLE_HF_TRANSFER=1)
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# Use the CLI to download all files for a specific model
RUN hf download maya-research/maya1 --local-dir .

FROM python:3.11-slim AS snac_downloader
WORKDIR /tmp/snac
# Install git-lfs if needed for large files
RUN apt-get update && apt-get install -y git-lfs && git lfs install
# Install huggingface-cli
RUN pip install huggingface-hub
# Use hf-transfer for faster downloads if desired (set HF_HUB_ENABLE_HF_TRANSFER=1)
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# Use the CLI to download all files for a specific model
RUN hf download hubertsiuzdak/snac_24khz --local-dir .

# Stage 2: Final application image
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
# Copy the downloaded model from the first stage
COPY --from=model_downloader /tmp/model ./local_model
# Copy the downloaded snac encoder from the second step
COPY --from=snac_downloader /tmp/snac ./local_snac


# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY maya1/ maya1/
COPY server.sh .
COPY samples.txt .
COPY README.md .

# Create logs directory
RUN mkdir -p logs

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "maya1.api_v2:app", "--host", "0.0.0.0", "--port", "8000"]
