"""Run Manga Image Translator's text detector and inpainter without OCR or translation."""

import argparse
import asyncio
import os
import shutil
import sys

import cv2
import numpy as np
import torch


DETECTOR_SIZE = 1024
MIN_TILE_OVERLAP = 128
TILE_SCALE = 0.57  # Approximately two crops along a normal page's long edge.
PAGE_BOX_THRESHOLD = 0.60
TILE_BOX_THRESHOLD = 0.50
MASK_PADDING_AT_3120 = 4
OUTLINE_REACH_AT_3120 = 12
WHITE_CORE_SIZE_AT_3120 = 64
WHITE_MIN_CHANNEL = 205


def _tile_starts(length, size, overlap):
    if length <= size:
        return [0]
    span = length - size
    stride = size - overlap
    intervals = (span + stride - 1) // stride
    return [i * span // intervals for i in range(intervals + 1)]


async def _detect_mask(detector, rgb, config):
    async def detect(image, threshold):
        _, mask, _ = await detector.detect(
            image, DETECTOR_SIZE, config.text_threshold, threshold,
            config.unclip_ratio, invert=False, gamma_correct=False, rotate=False,
        )
        return mask

    mask = await detect(rgb, PAGE_BOX_THRESHOLD)
    h, w = rgb.shape[:2]
    # Keep a page-level pass for large text. Overlapping crops give small
    # lettering more detector pixels; scale the crops with large upscales to
    # avoid running dozens of nearly identical model passes.
    size = max(DETECTOR_SIZE, round(max(h, w) * TILE_SCALE))
    if max(h, w) < DETECTOR_SIZE + MIN_TILE_OVERLAP:
        return mask
    overlap = max(MIN_TILE_OVERLAP, size // 8)
    margin = overlap // 4
    for y in _tile_starts(h, size, overlap):
        for x in _tile_starts(w, size, overlap):
            bottom, right = min(y + size, h), min(x + size, w)
            crop = await detect(rgb[y:bottom, x:right], TILE_BOX_THRESHOLD)
            # Ignore crop edges unless they coincide with the image edge.
            top, left = y + (margin if y else 0), x + (margin if x else 0)
            end_y = bottom - (margin if bottom < h else 0)
            end_x = right - (margin if right < w else 0)
            dst = mask[top:end_y, left:end_x]
            cv2.bitwise_or(dst, crop[top - y:end_y - y, left - x:end_x - x], dst=dst)
    return mask


def _pad_mask(mask):
    radius = max(1, round(mask.shape[1] * MASK_PADDING_AT_3120 / 3120))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def _include_white_outline(rgb, padded_mask):
    """Include light pixels connected to detected strokes, not all nearby artwork."""
    scale = padded_mask.shape[1] / 3120
    reach = min(24, max(1, round(OUTLINE_REACH_AT_3120 * scale)))
    core_size = max(4, round(WHITE_CORE_SIZE_AT_3120 * scale))
    white = cv2.inRange(rgb, (WHITE_MIN_CHANNEL,) * 3, (255,) * 3)
    mask = cv2.compare(padded_mask, 127, cv2.CMP_GE)
    # Each growth step can cross only bright pixels directly touching the mask.
    # Bound the number of steps so adjoining paper and highlights cannot flood.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    count = cv2.countNonZero(mask)
    for _ in range(reach):
        grown = cv2.dilate(mask, kernel)
        cv2.bitwise_and(grown, white, dst=grown)
        cv2.bitwise_or(mask, grown, dst=mask)
        new_count = cv2.countNonZero(mask)
        if new_count == count:
            break
        count = new_count

    # Fill small enclosed, mostly white cores, including anti-aliased pixels
    # that fell below the brightness threshold. Ignore larger enclosed artwork.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
            if parent < 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) > core_size:
                continue
            core = np.zeros((h, w), np.uint8)
            cv2.drawContours(core, [contour], 0, 255, cv2.FILLED, offset=(-x, -y))
            region = mask[y:y + h, x:x + w]
            uncovered = (core != 0) & (region == 0)
            size = np.count_nonzero(uncovered)
            if size and np.count_nonzero(white[y:y + h, x:x + w][uncovered]) >= 0.8 * size:
                region[uncovered] = 255
    return mask


def _write_mask_overlay(path, bgr, padded_mask, mask):
    overlay = bgr.copy()
    for pixels, color in ((padded_mask >= 127, (0, 255, 0)),
                          ((mask != 0) & (padded_mask < 127), (255, 0, 255))):
        overlay[pixels] = overlay[pixels] // 2 + np.array(color, np.uint8) // 2
    if not cv2.imwrite(path, overlay):
        raise RuntimeError("Could not write the mask overlay")


async def clean(source, destination, mask_overlay=None):
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
    await detector.load(device, input_size=DETECTOR_SIZE)
    try:
        mask = await _detect_mask(detector, rgb, DetectorConfig())
    finally:
        await detector.unload()
    padded_mask = _pad_mask(mask)
    mask = _include_white_outline(rgb, padded_mask)
    if mask_overlay is not None:
        _write_mask_overlay(mask_overlay, bgr, padded_mask, mask)
    if not np.any(mask):
        shutil.copyfile(source, destination)
        return

    del bgr, padded_mask
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
    parser.add_argument("--mask-overlay", help="Save a debug overlay on the source image")
    args = parser.parse_args()
    asyncio.run(clean(args.source, args.destination, args.mask_overlay))
