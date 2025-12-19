# ARG for runtime type: "cpu" or "gpu"
ARG RUNTIME=cpu

# Base image selection
FROM python:3.10-slim AS base-cpu
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-gpu

# Select base based on RUNTIME arg
FROM base-${RUNTIME} AS final

# Re-declare ARG for this stage
ARG RUNTIME

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev \
    libgl1 libgomp1 libsm6 libxext6 libxrender1 curl \
    && if [ "$RUNTIME" = "gpu" ]; then \
         apt-get install -y --no-install-recommends python3.10 python3-pip && \
         ln -sf /usr/bin/python3 /usr/bin/python && \
         ln -sf /usr/bin/pip3 /usr/bin/pip; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install common dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Runtime-specific ONNX
RUN if [ "$RUNTIME" = "gpu" ]; then \
    echo "--- Installing GPU Support (onnxruntime-gpu) ---" && \
    pip install --no-cache-dir onnxruntime-gpu; \
    else \
    echo "--- Installing CPU Support (onnxruntime) ---" && \
    pip install --no-cache-dir onnxruntime; \
    fi

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser && \
    chown -R appuser:appgroup /app

USER appuser
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ===== Local build & run cheatsheet =====
# CPU build:
#   docker build --build-arg RUNTIME=cpu -t vowscan:cpu .
#
# CPU run (foreground):
#   docker run -p 8501:8501 vowscan:cpu
#
# CPU run (background + auto-restart):
#   docker run -d --restart unless-stopped -p 8501:8501 --name vowscan_cpu vowscan:cpu
#
# GPU build:
#   docker build --build-arg RUNTIME=gpu -t vowscan:gpu .
#
# GPU run (foreground, with GPU):
#   docker run --gpus all -p 8501:8501 vowscan:gpu
#
# GPU run (background + auto-restart):
#   docker run -d --gpus all --restart unless-stopped -p 8501:8501 --name vowscan_gpu vowscan:gpu

