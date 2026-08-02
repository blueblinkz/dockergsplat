FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9"  # ADA GPUs

# System deps
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python env
RUN pip3 install --upgrade pip

# Install PyTorch (CUDA 12)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install gsplat
RUN pip install gsplat

# Install nerfstudio
RUN pip install nerfstudio

# COLMAP (needed for nerfstudio pipelines)
RUN apt-get update && apt-get install -y colmap

WORKDIR /workspace

# Copy repo
COPY . /workspace

# Default entry
CMD ["bash"]
