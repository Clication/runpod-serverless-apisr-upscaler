"""
RunPod Serverless handler
APISR 4x -> optional repeated APISR 2x -> ImageMagick resize -> JPEG + AVIF

輸入:
    {"input": ["https://.../a.jpg", "https://.../b.png"]}
    {"input": {"images": ["https://.../a.jpg"]}}

輸出:
    {"0001": {"jpeg": "<b64>", "avif": "<b64>"}, "0002": {...}}
    （依輸入順序編號，切半時右半在先；失敗的編號記在 "errors" 下）
"""

import base64
import os
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass
from itertools import count
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import requests
import runpod
import torch

# ---------------------------------------------------------------- 設定
APISR_ROOT = os.environ.get("APISR_ROOT", "/app/APISR")
sys.path.insert(0, APISR_ROOT)
from architecture.rrdb import RRDBNet  # noqa: E402

IM_BIN = os.environ.get("IM_BIN", "convert")
WEIGHT_PATHS = {
    scale: os.environ.get(
        env_name,
        os.path.join(APISR_ROOT, "pretrained", f"{scale}x_APISR_RRDB_GAN_generator.pth"),
    )
    for scale, env_name in ((4, "APISR_WEIGHT"), (2, "APISR_WEIGHT_2X"))
}
INITIAL_SCALE = 4
EXTRA_SCALE = 2
TILE_OVERLAP = 16
DL_TIMEOUT = (10, 90)


@dataclass(frozen=True)
class JobOptions:
    OUT_WIDTH: int = 2048
    SPLIT_HORI_IMAGE_TO_HALF: bool = False
    PRE_MAX_SIDE: int = 0
    TILE: int = 0
    JPEG_QUALITY: int = 85
    AVIF_QUALITY: int = 60
    AVIF_SPEED: int = 6
    MAX_IMAGES: int = 20
    MAX_DOWNLOAD_BYTES: int = 128 * 1024 * 1024
    APISR_FP16: bool = True
    APISR_REPEAT_2X: bool = False

    @classmethod
    def from_input(cls, job_input):
        if not isinstance(job_input, dict):
            return cls()

        values = {}
        for name, default in vars(cls()).items():
            if name not in job_input:
                continue
            value = job_input[name]
            if isinstance(default, bool):
                if type(value) is not bool:
                    raise ValueError(f"{name} must be a JSON boolean")
            else:
                minimum, maximum = {
                    "PRE_MAX_SIDE": (0, None),
                    "TILE": (0, None),
                    "JPEG_QUALITY": (1, 100),
                    "AVIF_QUALITY": (1, 100),
                    "AVIF_SPEED": (0, 10),
                }.get(name, (1, None))
                if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
                    limits = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
                    raise ValueError(f"{name} must be a JSON integer {limits}")
            values[name] = value
        return cls(**values)

# ---------------------------------------------------------------- 模型
_generators = {}
_model_lock = threading.Lock()


def _create_generator(device, scale, use_fp16):
    weight_path = WEIGHT_PATHS[scale]
    if not os.path.exists(weight_path):
        raise RuntimeError(f"找不到權重檔: {weight_path}")

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "model_state_dict"):
        if key in checkpoint:
            weight = checkpoint[key]
            break
    else:
        if not checkpoint or not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            raise RuntimeError("Unsupported APISR checkpoint format")
        weight = checkpoint

    weight = {(key[10:] if key.startswith("_orig_mod.") else key): value for key, value in weight.items()}
    generator = RRDBNet(3, 3, scale=scale)
    generator.load_state_dict(weight)
    generator = generator.eval().to(device)
    if use_fp16 and device == "cuda":
        generator = generator.half()
    return generator


def get_generator(scale, use_fp16):
    """Load and cache RRDB models by scale and precision."""
    key = (scale, use_fp16)
    with _model_lock:
        if key in _generators:
            return _generators[key]

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run this worker on a RunPod NVIDIA GPU instance.")

        print(
            f"[apisr] torch={torch.__version__} CUDA={torch.version.cuda} "
            f"GPU={torch.cuda.get_device_name(0)}",
            flush=True,
        )
        generator = _create_generator("cuda", scale, use_fp16)

        torch.backends.cudnn.benchmark = True
        _generators[key] = generator
        print(f"[apisr] {scale}x RRDB 模型已載入 (fp16={use_fp16})", flush=True)
        return generator


# ---------------------------------------------------------------- 工具
def _extract_urls(job_input):
    if isinstance(job_input, list):
        return [u for u in job_input if isinstance(u, str) and u.strip()]
    if isinstance(job_input, str):
        return [job_input] if job_input.strip() else []
    if isinstance(job_input, dict):
        for key in ("images", "urls", "image_urls", "input"):
            value = job_input.get(key)
            if isinstance(value, list):
                return [u for u in value if isinstance(u, str) and u.strip()]
            if isinstance(value, str) and value.strip():
                return [value]
    return []


def _safe_name(url, index):
    path = unquote(urlparse(url).path)
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:240] or f"image{index + 1}"


def _download(url, dest, session, max_download_bytes):
    with session.get(url, timeout=DL_TIMEOUT, stream=True) as resp:
        resp.raise_for_status()
        total = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_download_bytes:
                    raise ValueError(f"檔案超過 {max_download_bytes} bytes 上限")
                fh.write(chunk)
    if total == 0:
        raise ValueError("下載到空檔案")
    return dest


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ImageMagick 失敗: {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def _b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


# ---------------------------------------------------------------- 推論
@torch.inference_mode()
def _super_resolve(generator, img_bgr, scale, options):
    height, width = img_bgr.shape[:2]
    tile = options.TILE
    if scale == EXTRA_SCALE:
        # The 2x model uses pixel unshuffle, so the image and each tile
        # must have even dimensions and begin on an even pixel boundary.
        if height % 2 or width % 2:
            img_bgr = cv2.copyMakeBorder(img_bgr, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
        if tile:
            tile = max(2, tile - tile % 2)
    h, w = img_bgr.shape[:2]

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    dtype = torch.float16 if options.APISR_FP16 else torch.float32
    tensor = tensor.to(device="cuda", dtype=dtype)

    if tile and (h > tile or w > tile):
        out = torch.zeros((1, 3, h * scale, w * scale), dtype=dtype, device="cuda")
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                ey0, ex0 = max(0, y0 - TILE_OVERLAP), max(0, x0 - TILE_OVERLAP)
                ey1, ex1 = min(h, y1 + TILE_OVERLAP), min(w, x1 + TILE_OVERLAP)
                patch = tensor[:, :, ey0:ey1, ex0:ex1]
                tile_out = generator(patch)
                dy, dx = (y0 - ey0) * scale, (x0 - ex0) * scale
                out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = tile_out[
                    :, :, dy:dy + (y1 - y0) * scale, dx:dx + (x1 - x0) * scale
                ]
                del tile_out
    else:
        out = generator(tensor)

    out = out[:, :, :height * scale, :width * scale]
    out = out.clamp_(0, 1).mul_(255).round_().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    del tensor
    torch.cuda.empty_cache()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _encode(src_png, work_dir, base, options):
    jpeg_path = os.path.join(work_dir, base + ".jpg")
    avif_path = os.path.join(work_dir, base + ".avif")
    common = [
        IM_BIN, src_png,
        # "-auto-orient",
        # "-filter", "Lanczos",
        "-resize", f"{options.OUT_WIDTH}x",
        # "-depth", "8",
        # "-strip",
    ]
    _run(common + [
        "-background", "white", "-alpha", "remove", "-alpha", "off",
        "-sampling-factor", "4:2:0", "-interlace", "JPEG",
        "-quality", str(options.JPEG_QUALITY), jpeg_path,
    ])
    _run(common + [
        "-quality", str(options.AVIF_QUALITY),
        "-define", f"heic:speed={options.AVIF_SPEED}",
        avif_path,
    ])
    return jpeg_path, avif_path


def _image_parts(img, name, split_landscape):
    h, w = img.shape[:2]
    if split_landscape and w > h:
        mid = w // 2
        yield f"{name}_a", img[:, mid:]
        yield f"{name}_b", img[:, :mid]
    else:
        yield name, img


def _upscale(generator, img, options):
    sr = _super_resolve(generator, img, INITIAL_SCALE, options)
    if options.APISR_REPEAT_2X and sr.shape[1] <= options.OUT_WIDTH:
        repeat_generator = get_generator(EXTRA_SCALE, options.APISR_FP16)
        while sr.shape[1] <= options.OUT_WIDTH:
            width = sr.shape[1]
            sr = _super_resolve(repeat_generator, sr, EXTRA_SCALE, options)
            if sr.shape[1] <= width:
                raise RuntimeError("APISR 2x pass did not increase the image width")
    return sr


def _process_image(generator, img, work_dir, name, options):
    # 有 alpha 的話先鋪白底（AVIF/JPEG 統一輸出，避免 JPEG 出黑邊）
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        img = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 限制長邊，避免 4x 之後爆 VRAM
    if options.PRE_MAX_SIDE > 0:
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > options.PRE_MAX_SIDE:
            ratio = options.PRE_MAX_SIDE / long_side
            img = cv2.resize(
                img, (max(1, int(w * ratio)), max(1, int(h * ratio))),
                interpolation=cv2.INTER_AREA,
            )

    sr = _upscale(generator, img, options)
    sr_png = os.path.join(work_dir, f"{name}.png")
    if not cv2.imwrite(sr_png, sr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError("無法寫入超解析影像")
    del sr

    jpeg_path, avif_path = _encode(sr_png, work_dir, name, options)
    return {"jpeg": _b64(jpeg_path), "avif": _b64(avif_path)}


# ---------------------------------------------------------------- handler
def handler(job):
    job_input = job.get("input")
    try:
        options = JobOptions.from_input(job_input)
    except ValueError as exc:
        return {"error": str(exc)}
    urls = _extract_urls(job_input)
    if not urls:
        return {"error": "input 需為 URL 陣列，例如 {\"input\": { \"images\": [\"https://...\"]}}"}
    if len(urls) > options.MAX_IMAGES:
        print(f"[apisr] 收到 {len(urls)} 張，只處理前 {options.MAX_IMAGES} 張", flush=True)
        urls = urls[:options.MAX_IMAGES]

    generator = get_generator(INITIAL_SCALE, options.APISR_FP16)
    results = {}
    errors = {}
    positions = count(1)

    try:
        with requests.Session() as session:
            session.headers["User-Agent"] = "apisr-serverless/1.0"

            for index, url in enumerate(urls):
                name = _safe_name(url, index)
                with tempfile.TemporaryDirectory(prefix="apisr_") as work_dir:
                    src_path = os.path.join(work_dir, "source.img")
                    try:
                        _download(url, src_path, session, options.MAX_DOWNLOAD_BYTES)
                        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
                        if img is None:
                            raise ValueError("無法解碼影像")
                    except Exception as exc:  # noqa: BLE001
                        key = f"{next(positions):04d}"
                        errors[key] = f"{type(exc).__name__}: {exc}"
                        print(f"[apisr] FAIL {key} {name} <- {url}\n{traceback.format_exc()}", flush=True)
                        continue

                    for part_name, part_img in _image_parts(img, name, options.SPLIT_HORI_IMAGE_TO_HALF):
                        key = f"{next(positions):04d}"
                        try:
                            results[key] = _process_image(generator, part_img, work_dir, part_name, options)
                            print(f"[apisr] OK {key} {part_name} <- {url}", flush=True)
                        except Exception as exc:  # noqa: BLE001
                            errors[key] = f"{type(exc).__name__}: {exc}"
                            print(f"[apisr] FAIL {key} {part_name} <- {url}\n{traceback.format_exc()}", flush=True)
                    del img, part_img
    finally:
        torch.cuda.empty_cache()

    output = dict(results)
    if errors:
        output["errors"] = errors
    return output



def smoke_test():
    for scale in (INITIAL_SCALE, EXTRA_SCALE):
        generator = _create_generator("cpu", scale, False)
        with torch.inference_mode():
            output = generator(torch.zeros(1, 3, 8, 8))
        if output.shape != (1, 3, 8 * scale, 8 * scale) or not torch.isfinite(output).all():
            raise RuntimeError(f"APISR {scale}x inference smoke test failed")

    with tempfile.TemporaryDirectory(prefix="apisr_smoke_") as work_dir:
        source = os.path.join(work_dir, "source.png")
        if not cv2.imwrite(source, np.zeros((32, 32, 3), dtype=np.uint8)):
            raise RuntimeError("PNG encoding smoke test failed")
        for path in _encode(source, work_dir, "output", JobOptions(OUT_WIDTH=64)):
            if os.path.getsize(path) == 0:
                raise RuntimeError(f"Empty encoded output: {path}")
            _run([IM_BIN, path, "null:"])
    print("[apisr] CPU model inference and JPEG/AVIF smoke test passed", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke-test"]:
        smoke_test()
    else:
        get_generator(INITIAL_SCALE, JobOptions().APISR_FP16)
        runpod.serverless.start({"handler": handler})
