[![Runpod](https://api.runpod.io/badge/Clication/runpod-serverless-apisr-upscaler)](https://console.runpod.io/hub/listing/Clication/runpod-serverless-apisr-upscaler)

# APISR RunPod Serverless

APISR 4x RRDB super-resolution with JPEG and AVIF output, resized to 2048 pixels wide by default.

## CUDA 13.2

The image uses NVIDIA CUDA 13.2, cuDNN, Ubuntu 24.04, Python 3.12,
and PyTorch 2.12.0 from the official `cu132` wheel index. Deploy on a
RunPod NVIDIA GPU worker whose driver supports CUDA 13.2 and whose GPU
is supported by this PyTorch build. No host CUDA toolkit installation is needed.

The CUDA version shown by `nvidia-smi` is the maximum CUDA level supported
by the host driver, not a requirement that every container use that exact
version. Newer drivers can generally run older CUDA applications, but older
PyTorch builds may lack support for newer GPU architectures.

## Build And Deploy

```sh
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/apisr:cuda13.2 .
docker push YOUR_DOCKERHUB_USER/apisr:cuda13.2
```

Set your RunPod Serverless endpoint's container image to the pushed tag.
Keep the image's default command. Model weights are downloaded during the
build, so worker startup does not need to download them.

The build checks dependency consistency, the PyTorch CUDA version, real
checkpoint loading, small CPU inferences for both models, and JPEG/AVIF encoding and decoding.
These checks do not require a GPU and do not replace a GPU deployment test.

## POST Request Options

Put processing options alongside `images` inside the RunPod `input` object:

```json
{
  "input": {
    "images": ["https://YOUR_HOST/example.png"],
    "OUT_WIDTH": 2048,
    "APISR_REPEAT_2X": true,
    "SPLIT_HORI_IMAGE_TO_HALF": true,
    "PRE_MAX_SIDE": 768,
    "TILE": 256
  }
}
```

Options apply only to this job. Existing string and array forms of `input` still
work with defaults. Use JSON numbers and booleans, not quoted environment
values such as `"2048"`, `"Y"`, or `"1"`. Existing environment settings for these
options are no longer read. `APISR_ROOT`, `APISR_WEIGHT`, `APISR_WEIGHT_2X`, and
`IM_BIN` remain deployment-only.

| Option | Default | Meaning |
| --- | --- | --- |
| `OUT_WIDTH` | `2048` | Final width in pixels; ImageMagick preserves aspect ratio. |
| `APISR_REPEAT_2X` | `false` | After the 4× pass, apply the 2× APISR model while the current width is less than or equal to `OUT_WIDTH`. Then resize to exactly `OUT_WIDTH`. |
| `SPLIT_HORI_IMAGE_TO_HALF` | `false` | Split landscape images before upscaling, with the right half returned first. |
| `PRE_MAX_SIDE` | `0` | Cap the longest side before the initial APISR pass; `0` disables the cap. |
| `TILE` | `0` | Inference tile size in pixels; `0` disables tiling. The 2× pass uses an even tile size. |
| `APISR_FP16` | `true` | Use FP16 inference for both models; `false` uses FP32. |
| `JPEG_QUALITY` | `85` | JPEG quality, 1–100. |
| `AVIF_QUALITY` | `60` | AVIF quality, 1–100. |
| `AVIF_SPEED` | `6` | AVIF encoder speed, 0–10. |
| `MAX_IMAGES` | `20` | Maximum input URLs per job; extra URLs are ignored. Split halves do not count separately. |
| `MAX_DOWNLOAD_BYTES` | `134217728` | Maximum bytes downloaded per image. |

For example, if the initial 4× pass produces a width of 512 px and `OUT_WIDTH`
is 2048, repeat mode produces 1024, 2048, then 4096 px before ImageMagick
resizes to 2048 px. If the initial width already exceeds the target, the 2×
model is not loaded. With repeat mode off, ImageMagick resizes directly after
the 4× pass, even if that means resizing up.

## GPU Smoke Test

On a machine with an NVIDIA GPU and NVIDIA Container Toolkit, create a
`test_input.json` containing a publicly accessible image URL:

```json
{"input": {"images": ["https://YOUR_HOST/example.png"]}}
```

```sh
docker run --rm --gpus all -v "${PWD}/test_input.json:/app/test_input.json:ro" YOUR_DOCKERHUB_USER/apisr:cuda13.2
```

Startup logs report PyTorch, CUDA, and GPU versions. A successful job returns
base64 `jpeg` and `avif` strings under numbered keys (`"0001"`, `"0002"`, …)
in input order. For example, if the first image is split and the second is not,
`"0001"` is the right half, `"0002"` is the left half, and `"0003"` is the
second image. Failed images or halves keep their index under `errors` rather
than shifting later results.

The same input can be sent in a RunPod endpoint test request.

## Memory Settings

For large inputs or smaller GPUs, try `"PRE_MAX_SIDE": 768` and `"TILE": 256`
in the request. Tiling reduces intermediate inference memory but still allocates
the full output on the GPU. Each additional 2× pass produces four times as many
pixels, so repeat mode can increase processing time and peak memory.
