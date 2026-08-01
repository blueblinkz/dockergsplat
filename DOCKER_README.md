# Splat Pipeline Docker Image

Builds COLMAP, GLOMAP, and OpenSplat from source in one image, and pip-installs
`pycolmap-cuda12` and `gsplat` alongside them.

## Build

```bash
docker build -t splat-pipeline:cu124 \
  --build-arg CUDA_ARCHITECTURES="86;89;90;120" \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg TORCH_VERSION=2.4.1 \
  --build-arg TORCH_CUDA_TAG=cu124 \
  .
```

Expect 30-60+ minutes on first build (COLMAP and OpenImageIO are the slow parts).
Nothing here uses BuildKit cache mounts, so re-runs after a Dockerfile edit will
rebuild every layer after the change — add `--mount=type=cache` to the `ninja`
steps yourself if you're iterating a lot.

## Run

```bash
docker run --gpus all -it -v /workspace:/workspace splat-pipeline:cu124
```

Inside the container: `colmap`, `glomap`, and `opensplat` are on `PATH`.
Python (`/opt/venv/bin/python`, also first on `PATH`) has `pycolmap`, `gsplat`,
and `torch` importable.

## Keeping CUDA/PyTorch versions in sync

The three build args `CUDA_VERSION` (base image), `TORCH_VERSION` +
`TORCH_CUDA_TAG` (LibTorch download for OpenSplat's C++ build), and the pip
`torch` install must all agree, or you'll get silent CPU fallback or a link
error. Check available LibTorch CUDA tags before changing `CUDA_VERSION`:
https://download.pytorch.org/libtorch/ — the folder names (`cu121`, `cu124`,
`cu126`, ...) are the valid `TORCH_CUDA_TAG` values.

## GPU architecture reference

Covers the GPU fleet actually rented on RunPod (A4500, RTX 2000 Ada, L4,
L40S, RTX 4090, RTX 5090), plus Hopper (90) included pre-emptively in case
an H100 gets rented later. Drop 90 if you never touch Hopper — each extra
arch adds to CUDA kernel compile time for COLMAP/OpenSplat.

| GPU | Compute capability | Arch flag |
|---|---|---|
| A4500 | Ampere | 86 |
| RTX 2000 Ada / L4 / L40S / RTX 4090 | Ada Lovelace | 89 |
| H100 | Hopper | 90 |
| RTX 5090 | Blackwell | 120 |

`CMAKE_CUDA_ARCHITECTURES` (COLMAP/GLOMAP/OpenSplat) and `TORCH_CUDA_ARCH_LIST`
(PyTorch/gsplat JIT kernels) are both set from the `CUDA_ARCHITECTURES` build
arg via the Dockerfile's `ENV` line — trim the list to the GPUs you actually
target to cut build time, since each extra arch roughly multiplies CUDA kernel
compile time for COLMAP and OpenSplat.

## Things that bit us building this on RunPod, encoded into the Dockerfile

- **OpenImageIO before COLMAP.** COLMAP's CMakeLists hard-requires OpenImageIO
  via `find_package`; if it's missing, some COLMAP builds fail outright and
  others silently disable features. It's built and `ninja install`ed first.
- **GLOMAP needs CMake ≥3.28.** Ubuntu 22.04's apt `cmake` is older, so a
  fresh CMake is installed into a venv that's placed first on `PATH` for the
  whole build.
- **`pycolmap-cuda12` doesn't need the from-source COLMAP build.** It's a
  self-contained wheel with bundled CUDA runtime libs — it's independent of
  the `colmap`/`glomap` CLI binaries, just pip-installed for convenience.
- **GLOMAP is upstream-deprecated.** The `colmap/glomap` repo is currently
  tagged `[DEPRECATED]` on GitHub as of this writing but still builds and runs
  against current COLMAP; worth periodically checking if colmap.github.io
  points you to a successor.
- **Nothing needs `/workspace` redirection here** the way your RunPod
  `setsplat.sh` does — everything installs to `/usr/local` inside the image
  layer itself, which persists as part of the image. `/workspace` is only
  mounted in for your actual data/output.

## If a GitHub `main` branch build breaks

Pin `COLMAP_GIT_COMMIT`, `GLOMAP_GIT_COMMIT`, or `OPENSPLAT_GIT_COMMIT` to a
known-good commit hash as a `--build-arg` rather than tracking `main`.
