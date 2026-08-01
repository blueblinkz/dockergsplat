# ==============================================================================
# Gaussian Splatting Pipeline: COLMAP + OpenImageIO + GLOMAP + OpenSplat
#                               + pycolmap-cuda12 + gsplat
#
# Build:
#   docker build -t splat-pipeline:cu124 \
#     --build-arg CUDA_ARCHITECTURES="86;89;90;120" .
#
#   CUDA_ARCHITECTURES covers: 86=Ampere (A4500), 89=Ada Lovelace
#   (RTX 2000 Ada, L4, L40S, RTX 4090), 90=Hopper (H100, included in case
#   you rent one), 120=Blackwell (RTX 5090). Trim this list to speed up
#   the build if you drop a GPU family.
#
# Run:
#   docker run --gpus all -it -v /workspace:/workspace splat-pipeline:cu124
#
# Notes:
#   - GLOMAP's upstream repo is marked [DEPRECATED] by the colmap org as of
#     2026, but it still builds and runs fine against modern COLMAP. Confirm
#     it still meets your needs before relying on it long-term.
#   - OpenImageIO is built and installed BEFORE COLMAP because COLMAP's
#     CMakeLists has a hard find_package(OpenImageIO) dependency that fails
#     silently if OIIO isn't already on the CMake prefix path.
#   - This mirrors the gotchas from setsplat.sh: build artifacts belong under
#     a path you intend to keep (here, everything installs to /usr/local
#     inside the image itself, so no /workspace redirection is needed -
#     unlike a bare RunPod pod, the image layer IS persistent storage).
# ==============================================================================

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CUDA_ARCHITECTURES="86;89;90;120"
ARG TORCH_CUDA_TAG=cu124
ARG TORCH_VERSION=2.4.1
ARG OIIO_VERSION=v2.5.9.0
ARG COLMAP_GIT_COMMIT=main
ARG GLOMAP_GIT_COMMIT=main
ARG OPENSPLAT_GIT_COMMIT=main
ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0" \
    PATH=/opt/venv/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH}

# ------------------------------------------------------------------------------
# 1. System packages (COLMAP/GLOMAP build deps + OpenImageIO build deps + OpenCV)
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip pkg-config \
    build-essential ninja-build \
    python3 python3-pip python3-venv python3-dev \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-filesystem-dev libboost-thread-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev \
    libopencv-dev \
    libopenexr-dev libtiff-dev libpng-dev libjpeg-turbo8-dev libwebp-dev \
    libraw-dev libssl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Modern CMake (Ubuntu 22.04's apt cmake is too old for GLOMAP's >=3.28 requirement)
# and ninja, installed into a venv that stays first on PATH for the whole build.
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip "cmake>=3.28" ninja

WORKDIR /opt/src

# ------------------------------------------------------------------------------
# 2. OpenImageIO — MUST be built and installed before COLMAP
# ------------------------------------------------------------------------------
RUN git clone --depth 1 --branch ${OIIO_VERSION} \
    https://github.com/AcademySoftwareFoundation/OpenImageIO.git oiio && \
    cd oiio && mkdir build && cd build && \
    cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOIIO_BUILD_TESTS=OFF \
      -DOIIO_BUILD_TOOLS=ON \
      -DUSE_PYTHON=OFF && \
    ninja && ninja install && \
    cd /opt/src && rm -rf oiio

# ------------------------------------------------------------------------------
# 3. COLMAP (built with CUDA support)
# ------------------------------------------------------------------------------
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && git checkout ${COLMAP_GIT_COMMIT} && \
    mkdir build && cd build && \
    cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DCUDA_ENABLED=ON && \
    ninja && ninja install && \
    cd /opt/src && rm -rf colmap

# ------------------------------------------------------------------------------
# 4. GLOMAP (depends on the COLMAP install above)
# ------------------------------------------------------------------------------
RUN git clone https://github.com/colmap/glomap.git && \
    cd glomap && git checkout ${GLOMAP_GIT_COMMIT} && \
    mkdir build && cd build && \
    cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" && \
    ninja && ninja install && \
    cd /opt/src && rm -rf glomap

# ------------------------------------------------------------------------------
# 5. LibTorch (version pinned to match TORCH_CUDA_TAG) — needed for OpenSplat's
#    C++ build. Keep TORCH_VERSION/TORCH_CUDA_TAG in sync with the base image's
#    CUDA_VERSION; mismatches are the #1 cause of OpenSplat CUDA build failures.
# ------------------------------------------------------------------------------
RUN wget -q https://download.pytorch.org/libtorch/${TORCH_CUDA_TAG}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${TORCH_CUDA_TAG}.zip \
      -O libtorch.zip && \
    unzip -q libtorch.zip -d /opt && \
    rm libtorch.zip

# ------------------------------------------------------------------------------
# 6. OpenSplat
# ------------------------------------------------------------------------------
RUN git clone https://github.com/pierotofy/OpenSplat.git opensplat && \
    cd opensplat && git checkout ${OPENSPLAT_GIT_COMMIT} && \
    mkdir build && cd build && \
    cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DGPU_RUNTIME=CUDA && \
    ninja && \
    cp opensplat /usr/local/bin/ && \
    cd /opt/src && rm -rf opensplat

# ------------------------------------------------------------------------------
# 7. Python side: pycolmap-cuda12 + gsplat + matching PyTorch wheel
#    (pycolmap-cuda12 ships its own bundled CUDA runtime libs, so it does NOT
#    need the COLMAP build above — the two are independent, both pip-installed
#    here to save you re-implementing pipeline glue in C++.)
# ------------------------------------------------------------------------------
RUN /opt/venv/bin/pip install --no-cache-dir \
      torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} && \
    /opt/venv/bin/pip install --no-cache-dir \
      pycolmap-cuda12 \
      gsplat \
      opencv-python-headless \
      numpy \
      Pillow \
      tqdm

WORKDIR /workspace

# Quick sanity check baked into the image build (informational only, non-fatal —
# GPU isn't available at build time so torch.cuda.is_available() will read False here)
RUN (colmap -h > /dev/null 2>&1 || true) && \
    (glomap -h > /dev/null 2>&1 || true) && \
    (opensplat --help > /dev/null 2>&1 || true) && \
    /opt/venv/bin/python -c "import pycolmap, gsplat, torch; print('pycolmap', pycolmap.__version__); print('torch built with CUDA:', torch.version.cuda)"

CMD ["/bin/bash"]
