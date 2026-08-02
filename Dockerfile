# ------------------------------------------------------------------------------
# Minimal CUDA + PyTorch + gsplat + nerfstudio
# ------------------------------------------------------------------------------

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------------------
# Build args (kept from your original for flexibility)
# ------------------------------------------------------------------------------
ARG TORCH_VERSION=2.4.1
ARG TORCH_CUDA_TAG=cu124
ARG CUDA_ARCHITECTURES="89"

# Python deps
ARG NUMPY_PKG="numpy"
ARG PIL_PKG="pillow"
ARG OPENCV_PKG="opencv-python-headless"
ARG TQDM_PKG="tqdm"

# Core libs
ARG GSPLAT_PKG="gsplat"
ARG NERFSTUDIO_PKG="nerfstudio"

# ------------------------------------------------------------------------------
# System deps (minimal)
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# ------------------------------------------------------------------------------
# Install PyTorch (CUDA matched)
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${TORCH_CUDA_TAG} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}

# ------------------------------------------------------------------------------
# Install gsplat + nerfstudio + deps
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    ${GSPLAT_PKG} \
    ${NERFSTUDIO_PKG} \
    ${NUMPY_PKG} \
    ${PIL_PKG} \
    ${OPENCV_PKG} \
    ${TQDM_PKG} \
    imageio imageio-ffmpeg scikit-image lpips rich tyro

# ------------------------------------------------------------------------------
# Optional: tiny-cuda-nn (performance boost for nerfstudio)
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch || true

# ------------------------------------------------------------------------------
# CUDA tuning for ADA GPUs
# ------------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES}

# ------------------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------------------
RUN python -c "import torch; print('Torch:', torch.__version__)"
RUN python -c "import gsplat; print('gsplat OK')"
RUN python -c "import nerfstudio; print('nerfstudio OK')"

WORKDIR /workspace

CMD ["bash"]
