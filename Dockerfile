# Multi-stage Dockerfile (Ada-only) for Gaussian Splatting pipeline (COLMAP + OIIO + GLOMAP + OpenSplat + Python)
# Target: Ada Lovelace GPUs (SM 8.9)
# Usage example:
# DOCKER_BUILDKIT=1 docker build --build-arg CUDA_VERSION=12.4.1 --build-arg UBUNTU_VERSION=22.04 \
#   --build-arg CUDA_ARCHITECTURES="89" --build-arg TORCH_CUDA_ARCH_LIST="8.9" \
#   --build-arg TORCH_VERSION=2.4.1 --build-arg TORCH_CUDA_TAG=cu124 -t splat-pipeline:ada .

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.4.1
ARG CUDA_ARCHITECTURES="89"
ARG TORCH_CUDA_TAG=cu124
ARG TORCH_VERSION=2.4.1

# Pin source refs for reproducible builds (override via --build-arg if desired)
ARG OIIO_VERSION=v2.5.9.0
ARG COLMAP_GIT_REF=main
ARG GLOMAP_GIT_REF=main
ARG OPENSPLAT_GIT_REF=main

ARG PYTHON_VERSION=3.10
ARG PYCOLMAP_PKG="pycolmap-cuda12"
ARG GSPLAT_PKG="gsplat"
ARG OPENCV_PKG="opencv-python-headless==4.8.1.78"
ARG NUMPY_PKG="numpy==1.26.4"
ARG PIL_PKG="Pillow==10.0.0"
ARG TQDM_PKG="tqdm==4.66.1"
ARG VALIDATE_TORCH=true

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# ARGs declared before the first FROM are NOT visible inside a build stage
# unless re-declared here. Without this block, every ${...} reference below
# (OIIO_VERSION, CUDA_ARCHITECTURES, COLMAP_GIT_REF, etc.) silently evaluates
# to an empty string, which is what broke the `git clone --branch` command.
ARG CUDA_VERSION
ARG CUDA_ARCHITECTURES
ARG TORCH_CUDA_TAG
ARG TORCH_VERSION
ARG OIIO_VERSION
ARG COLMAP_GIT_REF
ARG GLOMAP_GIT_REF
ARG OPENSPLAT_GIT_REF
ARG PYCOLMAP_PKG
ARG GSPLAT_PKG
ARG OPENCV_PKG
ARG NUMPY_PKG
ARG PIL_PKG
ARG TQDM_PKG
ARG VALIDATE_TORCH

LABEL org.opencontainers.image.source="https://github.com/blueblinkz/dockergsplat"
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ca-certificates build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev \
    cmake ninja-build \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-filesystem-dev libboost-thread-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev libqt5svg5-dev \
    libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev \
    libopencv-dev \
    libopenexr-dev libtiff-dev libpng-dev libjpeg-turbo8-dev libwebp-dev \
    libraw-dev libssl-dev zlib1g && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip "cmake>=3.28" "ninja>=1.11"

# Python packages (venv) — done early, right after venv creation, so a broken
# pip install / torch-CUDA mismatch fails in minutes rather than after the
# ~2+ hour OIIO/COLMAP/GLOMAP/OpenSplat builds below. None of those C++
# builds depend on this step (OpenSplat links the C++ LibTorch downloaded
# separately below, not this pip-installed python torch package).
#
# NOTE: pycolmap-cuda12/gsplat pull in an older torch (observed: torch-2.0.1)
# with nvidia-*-cu11 dependencies. NVIDIA's split cu11/cu12 pip packages
# install their .so files to the SAME path inside site-packages regardless
# of package name, so whichever installs last silently overwrites the other's
# shared library on disk — this clobbered our cu124 libnccl.so.2 with an
# older cu11 build lacking symbols current torch needs (ncclCommRegister).
# Fix: purge the stray cu11 packages, then force-reinstall torch WITH its
# full dependency tree (no --no-deps) so the correct cu12 files come back.
RUN /opt/venv/bin/pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} && \
    /opt/venv/bin/pip install --no-cache-dir ${PYCOLMAP_PKG} ${GSPLAT_PKG} ${OPENCV_PKG} ${NUMPY_PKG} ${PIL_PKG} ${TQDM_PKG} && \
    /opt/venv/bin/pip uninstall -y \
      nvidia-cublas-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 \
      nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 nvidia-cufft-cu11 \
      nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 \
      nvidia-nccl-cu11 nvidia-nvtx-cu11 || true && \
    /opt/venv/bin/pip install --no-cache-dir --force-reinstall "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}

# Optional torch/CUDA validation (single RUN to avoid Dockerfile parser heredoc issues)
RUN if [ "${VALIDATE_TORCH}" = "true" ]; then \
      echo "import sys, torch" > /tmp/check_torch.py && \
      echo "expected = '${CUDA_VERSION}'.split('.')[:2]" >> /tmp/check_torch.py && \
      echo "cuda_ver = torch.version.cuda or ''" >> /tmp/check_torch.py && \
      echo "if not cuda_ver.startswith('{}.{}'.format(expected[0], expected[1])):" >> /tmp/check_torch.py && \
      echo "    print('ERROR: torch.version.cuda =', cuda_ver, 'does not start with expected', '{}.{}'.format(expected[0], expected[1]))" >> /tmp/check_torch.py && \
      echo "    sys.exit(1)" >> /tmp/check_torch.py && \
      echo "print('Torch CUDA check ok:', cuda_ver)" >> /tmp/check_torch.py && \
      python3 /tmp/check_torch.py && rm -f /tmp/check_torch.py; \
    fi

WORKDIR /opt/src

# OpenImageIO (shallow clone by tag)
RUN git clone --depth 1 --branch ${OIIO_VERSION} https://github.com/AcademySoftwareFoundation/OpenImageIO.git oiio && \
    cd oiio && mkdir -p build && cd build && \
    /opt/venv/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOIIO_BUILD_TESTS=OFF -DOIIO_BUILD_TOOLS=ON -DUSE_PYTHON=OFF && \
    /opt/venv/bin/ninja && /opt/venv/bin/ninja install && cd /opt/src && rm -rf oiio

# COLMAP
RUN git clone --depth 1 --branch ${COLMAP_GIT_REF} https://github.com/colmap/colmap.git && \
    cd colmap && mkdir -p build && cd build && \
    /opt/venv/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DCUDA_ENABLED=ON && \
    /opt/venv/bin/ninja && /opt/venv/bin/ninja install && cd /opt/src && rm -rf colmap

# GLOMAP
RUN git clone --depth 1 --branch ${GLOMAP_GIT_REF} https://github.com/colmap/glomap.git && \
    cd glomap && mkdir -p build && cd build && \
    /opt/venv/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" && \
    /opt/venv/bin/ninja && /opt/venv/bin/ninja install && cd /opt/src && rm -rf glomap

# LibTorch download and unpack
RUN set -eux; \
    TORCH_ZIP="libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_CUDA_TAG}.zip"; \
    URL="https://download.pytorch.org/libtorch/${TORCH_CUDA_TAG}/${TORCH_ZIP}"; \
    wget -q "$URL" -O /tmp/libtorch.zip && unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip

# OpenSplat build
RUN git clone --depth 1 --branch ${OPENSPLAT_GIT_REF} https://github.com/pierotofy/OpenSplat.git opensplat && \
    cd opensplat && mkdir -p build && cd build && \
    /opt/venv/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DGPU_RUNTIME=CUDA && \
    /opt/venv/bin/ninja && cp opensplat /usr/local/bin/ && cd /opt/src && rm -rf opensplat

# Shrink the image before export: cmake/ninja were only needed to build
# COLMAP/GLOMAP/OpenSplat above, not to run them - drop them from the venv
# that gets copied into the runtime stage. Also strip debug symbols from
# compiled libraries (a well-known way to cut real size off CUDA/PyTorch
# .so files) and clear caches, since export ran out of disk last time.
RUN /opt/venv/bin/pip uninstall -y cmake ninja || true && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    rm -rf /root/.cache/pip /opt/venv/pip-cache 2>/dev/null || true && \
    find /usr/local/lib /opt/libtorch/lib /opt/venv/lib -name "*.so*" -exec strip --strip-unneeded {} \; 2>/dev/null || true

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG USERNAME=splat
ARG USER_UID=1000
ARG USER_GID=1000

ENV PATH=/opt/venv/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-venv python3-pip \
    libjpeg-turbo8 libpng16-16 libtiff5 libwebp-dev libopenexr-dev libraw-dev \
    libssl3 zlib1g libsqlite3-0 libgomp1 libstdc++6 \
    libsm6 libice6 libxext6 libxrender1 libgl1 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} && mkdir -p /workspace && chown ${USERNAME}:${USERNAME} /workspace

WORKDIR /workspace

COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /opt/libtorch /opt/libtorch
COPY --from=builder /opt/venv /opt/venv

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig || true
RUN chown -R root:root /usr/local/bin /usr/local/lib /opt/libtorch && chmod -R a+rX /usr/local/lib /opt/libtorch

USER ${USERNAME}
ENV HOME=/home/${USERNAME}

# Non-fatal quick checks
RUN (command -v colmap >/dev/null 2>&1 && colmap -h >/dev/null 2>&1) || true
RUN (command -v opensplat >/dev/null 2>&1 && opensplat --help >/dev/null 2>&1) || true
RUN python3 -c "import sys; import gsplat, pycolmap, torch; print('torch.cuda:', torch.version.cuda)" || true

CMD ["/bin/bash"]
