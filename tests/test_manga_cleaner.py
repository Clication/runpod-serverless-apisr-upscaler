import unittest
from types import SimpleNamespace

import numpy as np

import manga_cleaner


class RecordingDetector:
    def __init__(self):
        self.calls = []

    async def detect(self, image, size, text_threshold, box_threshold, unclip_ratio, **kwargs):
        shape = image.shape[:2]
        self.calls.append((shape, size, box_threshold))
        return [], np.full(shape, 255, dtype=np.uint8), None


class MangaCleanerTests(unittest.IsolatedAsyncioTestCase):
    config = SimpleNamespace(text_threshold=0.5, unclip_ratio=2.3)

    async def test_small_image_uses_only_whole_image_detection(self):
        detector = RecordingDetector()
        mask = await manga_cleaner._detect_mask(detector, np.zeros((600, 800, 3), np.uint8), self.config)
        self.assertEqual(detector.calls, [((600, 800), 1024, manga_cleaner.PAGE_BOX_THRESHOLD)])
        self.assertTrue(np.all(mask == 255))

    async def test_overlapping_crops_cover_entire_image_after_trimming_edges(self):
        class TileOnlyDetector(RecordingDetector):
            async def detect(self, *args, **kwargs):
                lines, mask, refined = await super().detect(*args, **kwargs)
                if len(self.calls) == 1:
                    mask[:] = 0
                return lines, mask, refined

        for shape in ((1800, 1272), (1272, 1800), (2400, 500), (4415, 3120)):
            with self.subTest(shape=shape):
                detector = TileOnlyDetector()
                mask = await manga_cleaner._detect_mask(detector, np.zeros((*shape, 3), np.uint8), self.config)
                self.assertGreater(len(detector.calls), 1)
                self.assertTrue(all(call[2] == manga_cleaner.TILE_BOX_THRESHOLD for call in detector.calls[1:]))
                self.assertTrue(np.all(mask == 255))
                if shape == (4415, 3120):
                    self.assertEqual(len(detector.calls), 5)

    def test_dilation_scales_with_width_and_does_not_change_the_source_mask(self):
        for width in (1560, 3120, 6240):
            mask = np.zeros((101, width), np.uint8)
            mask[50, 100] = 255
            padded = manga_cleaner._pad_mask(mask)
            radius = round(width * manga_cleaner.MASK_PADDING_AT_3120 / 3120)
            self.assertEqual(np.count_nonzero(mask), 1)
            self.assertEqual(np.count_nonzero(padded[50]), radius * 2 + 1)
