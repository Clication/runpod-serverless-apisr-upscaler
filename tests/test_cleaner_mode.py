import base64
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import handler


class CleanerModeTests(unittest.TestCase):
    def test_cleaner_type_validation(self):
        self.assertEqual(handler.JobOptions.from_input({}).cleanerType, "panelCleaner")
        for backend in ("panelCleaner", "mangaImageTranslator"):
            self.assertEqual(handler.JobOptions.from_input({"cleanerType": backend}).cleanerType, backend)
        for invalid in ("unknown", "panelcleaner", True, None, []):
            with self.subTest(invalid=invalid):
                self.assertEqual(
                    handler.handler({"input": {"cleanerType": invalid}}),
                    {"error": 'cleanerType must be "panelCleaner" or "mangaImageTranslator"'},
                )

    def test_requires_json_boolean(self):
        self.assertFalse(handler.JobOptions.from_input(["https://example.com/image.png"]).cleanerMode)
        self.assertFalse(handler.JobOptions.from_input({"cleanerMode": False}).cleanerMode)
        self.assertTrue(handler.JobOptions.from_input({"cleanerMode": True}).cleanerMode)
        for value in ("true", 1, None):
            with self.subTest(value=value):
                self.assertEqual(
                    handler.handler({"input": {"cleanerMode": value}}),
                    {"error": "cleanerMode must be a JSON boolean"},
                )

    def test_cleaner_only_changes_avif_source(self):
        upscaled = np.zeros((20, 32, 3), dtype=np.uint8)

        def encode(cmd):
            self.assertEqual(cmd[2:4], ["-resize", "123x"])
            Path(cmd[-1]).write_bytes(Path(cmd[1]).read_bytes())

        def clean(src, work_dir):
            self.assertEqual(Path(src).read_bytes(), original_png)
            cleaned = Path(work_dir) / "cleaned.png"
            cleaned.write_bytes(b"cleaned image")
            return str(cleaned)

        with tempfile.TemporaryDirectory() as work_dir:
            original_path = Path(work_dir) / "reference.png"
            handler.cv2.imwrite(str(original_path), upscaled, [handler.cv2.IMWRITE_PNG_COMPRESSION, 1])
            original_png = original_path.read_bytes()
            with (
                patch.object(handler, "_upscale", return_value=upscaled),
                patch.object(handler, "_run", side_effect=encode),
                patch.object(handler, "_clean_png", side_effect=clean) as panel,
                patch.object(handler, "_clean_manga_png", side_effect=clean) as manga,
            ):
                for backend in ("panelCleaner", "mangaImageTranslator"):
                    for enabled in (False, True):
                        with self.subTest(backend=backend, enabled=enabled):
                            panel.reset_mock()
                            manga.reset_mock()
                            options = handler.JobOptions(OUT_WIDTH=123, cleanerMode=enabled, cleanerType=backend)
                            result = handler._process_image(None, upscaled, work_dir, "page", options)
                            self.assertEqual(set(result), {"jpeg", "avif"})
                            self.assertEqual(base64.b64decode(result["jpeg"]), original_png)
                            self.assertEqual(
                                base64.b64decode(result["avif"]), b"cleaned image" if enabled else original_png
                            )
                            self.assertEqual(panel.call_count, int(enabled and backend == "panelCleaner"))
                            self.assertEqual(manga.call_count, int(enabled and backend == "mangaImageTranslator"))

    def test_cleaner_rejects_missing_output(self):
        with tempfile.TemporaryDirectory() as work_dir, patch.object(handler.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            with self.assertRaisesRegex(RuntimeError, "did not export a cleaned PNG"):
                handler._clean_png(str(Path(work_dir) / "upscaled.png"), work_dir)
            with self.assertRaisesRegex(RuntimeError, "Manga Image Translator did not export a cleaned PNG"):
                handler._clean_manga_png(str(Path(work_dir) / "upscaled.png"), work_dir)

    def test_manga_cleaner_reports_subprocess_failure(self):
        with tempfile.TemporaryDirectory() as work_dir, patch.object(handler.subprocess, "run") as run:
            source = str(Path(work_dir) / "upscaled.png")
            run.return_value = subprocess.CompletedProcess([], 1, "", "inpainting failed")
            with self.assertRaisesRegex(RuntimeError, r"Manga Image Translator failed \(exit 1\): inpainting failed"):
                handler._clean_manga_png(source, work_dir)
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[:3], [handler.sys.executable, handler.MANGA_CLEANER_SCRIPT, source])
            self.assertEqual(Path(cmd[-1]).parent.parent, Path(work_dir))
