# https://share.google/aimode/d4j5jVuv6pkVPjY7y

# Stage 1: Fast Model Downloader
FROM python:3.11-slim AS downloader
WORKDIR /models
RUN pip install --no-cache-dir huggingface-hub hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Download both models in one stage to simplify
RUN hf download maya-research/maya1 --local-dir ./maya1
RUN hf download hubertsiuzdak/snac_24khz --local-dir ./snac

# Stage 2: Final Production Image
# Use official vLLM image for pre-compiled kernels optimized for Ada (L4)
FROM vllm/vllm-openai:latest

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Copy model artifacts from downloader stage
COPY --from=downloader /models/maya1 ./local_model
COPY --from=downloader /models/snac ./local_snac

# Create a virtual environment
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV

# Add the virtual environment's bin directory to the PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy application code and requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY maya1/ maya1/
COPY server.sh .

# Optimized for NVIDIA L4 (Compute Capability 8.9)
ENV VLLM_TORCH_CUDA_ARCH_LIST="8.9"
ENV HF_HUB_OFFLINE=1 

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Reset the inherited ENTRYPOINT from vllm-openai
ENTRYPOINT []

# Run the application
CMD ["python3", "-m", "uvicorn", "maya1.api_v2:app", "--host", "0.0.0.0", "--port", "8000"]

