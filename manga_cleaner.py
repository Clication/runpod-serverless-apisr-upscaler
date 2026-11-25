"""Run Manga Image Translator's text detector and inpainter without OCR or translation."""

import argparse
import asyncio
import os
import shutil
import sys

import cv2
import numpy as np
import torch


async def clean(source, destination):
    # The container ships only the upstream modules needed for text removal.
    sys.path.insert(0, os.environ.get("MANGA_TRANSLATOR_ROOT", "/app/manga-image-translator"))
    from manga_translator.config import DetectorConfig, InpainterConfig, InpaintPrecision
    from manga_translator.detection.ctd import ComicTextDetector
    from manga_translator.inpainting.inpainting_lama_mpe import LamaLargeInpainter

    bgr = cv2.imread(source, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not read the upscaled PNG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = ComicTextDetector()
    if not detector.is_downloaded():
        raise RuntimeError("Manga Image Translator detector weights are missing")
    await detector.load(device)
    config = DetectorConfig()
    try:
        _, mask, _ = await detector.detect(
            rgb, config.detection_size, config.text_threshold, config.box_threshold,
            config.unclip_ratio, invert=False, gamma_correct=False, rotate=False,
        )
    finally:
        await detector.unload()
    if not np.any(mask):
        shutil.copyfile(source, destination)
        return

    torch.cuda.empty_cache()
    inpainter = LamaLargeInpainter()
    if not inpainter.is_downloaded():
        raise RuntimeError("Manga Image Translator inpainting weights are missing")
    await inpainter.load(device)
    precision = InpaintPrecision.fp32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        precision = InpaintPrecision.bf16
    config = InpainterConfig(inpainting_precision=precision)
    cleaned = await inpainter.inpaint(rgb, mask, config, config.inpainting_size)
    if not cv2.imwrite(destination, cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Could not write the cleaned PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    asyncio.run(clean(args.source, args.destination))
