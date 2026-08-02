# CUDA + PyTorch + gsplat + nerfstudio

A minimal, single-stage image for running [gsplat](https://github.com/nerfstudio-project/gsplat) and [nerfstudio](https://docs.nerf.studio) on a GPU. No COLMAP/GLOMAP build included — bring your own preprocessed data (camera poses + sparse point cloud), or use nerfstudio's own `ns-process-data` command once inside the container.

## What's in the image

- CUDA 12.4.1 + cuDNN (runtime, not devel — no compiler toolchain, so this image can't build anything from source)
- PyTorch 2.4.1 (cu124), torchvision, torchaudio
- gsplat, nerfstudio
- numpy, pillow, opencv-python-headless, tqdm, imageio, imageio-ffmpeg, scikit-image, lpips, rich, tyro
- tiny-cuda-nn (best-effort — the build is wrapped in `|| true`, so the image still builds even if this step fails; check the build log to confirm it actually installed)
- ffmpeg, git

## Build

```bash
docker build -t gsplat-nerfstudio:ada \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg TORCH_VERSION=2.4.1 \
  --build-arg TORCH_CUDA_TAG=cu124 \
  --build-arg CUDA_ARCHITECTURES="89" \
  .
```

Build takes a few minutes — this is a pip-only image, nothing compiled from source except tiny-cuda-nn's CUDA extension.

## Run

```bash
docker run --gpus all -it -v /workspace:/workspace gsplat-nerfstudio:ada
```

Mount your data directory to `/workspace` so it's visible inside the container.

## Keeping versions in sync

`CUDA_VERSION`, `TORCH_VERSION`, and `TORCH_CUDA_TAG` must agree with each other, or torch will silently fall back to CPU or fail to import. The base image's CUDA runtime version and the `TORCH_CUDA_TAG` wheel index (`cu121`, `cu124`, `cu126`, ...) need to be from the same CUDA release family. Check available tags at https://download.pytorch.org/whl/torch/ before changing either.

## GPU architecture (`CUDA_ARCHITECTURES` / `TORCH_CUDA_ARCH_LIST`)

Defaults to `"89"` — Ada Lovelace only (RTX 4090, RTX 2000 Ada, L4, L40S). This value is used directly as `TORCH_CUDA_ARCH_LIST`, which controls which GPU architectures tiny-cuda-nn's CUDA extension compiles for.

| GPU | Architecture | Value |
|---|---|---|
| A4500, A100 | Ampere | `86` |
| RTX 4090, RTX 2000 Ada, L4, L40S | Ada Lovelace | `89` |
| H100 | Hopper | `90` |
| RTX 5090 | Blackwell | `120` — **needs CUDA ≥ 12.8**, won't compile against this image's CUDA 12.4.1 base |

For multiple architectures, use a semicolon-separated list, e.g. `"86;89"`. Note this Dockerfile passes the value straight through unquoted in one spot (`ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES}`) — that's fine as an `ENV` assignment (not parsed by a shell), but if you ever add a multi-value default to a `RUN` command in this file, quote it, or the shell will split on the semicolons.

## Sanity checks

The build itself verifies `torch`, `gsplat`, and `nerfstudio` all import cleanly (build fails if any of these three don't). It does **not** verify tiny-cuda-nn imports, since that install is allowed to fail silently. To confirm it actually installed, run inside the container:

```bash
python -c "import tinycudann; print('tiny-cuda-nn OK')"
```

## Known gaps

- No COLMAP, GLOMAP, or OpenSplat — this is a lighter-weight image for the gsplat/nerfstudio side only.
- Runtime (not devel) CUDA base — you can't compile new CUDA extensions inside a running container from this image; the tiny-cuda-nn build has to succeed at image-build time or not at all.
