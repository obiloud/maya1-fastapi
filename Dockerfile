# syntax=docker/dockerfile:1
# Stage 1: Fast Model Downloader
FROM python:3.11-slim AS downloader
WORKDIR /models

# Install hf_transfer for high-speed downloads
RUN pip install --no-cache-dir huggingface-hub hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Use BuildKit cache mounts to persist the HF cache across builds
# This ensures that even if the layer is invalidated, the files are already on disk
RUN --mount=type=cache,target=/root/.cache/huggingface \
    hf download maya-research/maya1 --local-dir ./maya1

RUN --mount=type=cache,target=/root/.cache/huggingface \
    hf download hubertsiuzdak/snac_24khz --local-dir ./snac

# Stage 2: Final Production Image
FROM vllm/vllm-openai:latest
WORKDIR /app

# 1. Install system dependencies (Rarely changes)
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Virtual Env (Static configuration)
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Optimized for NVIDIA L4 (Static configuration)
ENV VLLM_TORCH_CUDA_ARCH_LIST="8.9"
ENV HF_HUB_OFFLINE=1 
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# 4. Install Python dependencies (Changes occasionally)
# Copying only requirements.txt first to leverage cache
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt 

# 5. Copy model artifacts (Large but static)
COPY --from=downloader /models/maya1 ./local_model
COPY --from=downloader /models/snac ./local_snac

# 6. Copy application code (Changes frequently)
COPY server.sh .
COPY maya1/ maya1/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "maya1.api_v2:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]