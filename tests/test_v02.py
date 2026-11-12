"""v0.2 확정 기능 — 프리셋·회전·실패 폴더·이력 CSV·비율 경고."""
import csv
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import cli  # noqa: E402
from test_core import make_page  # noqa: E402


class TestPreset(unittest.TestCase):
    def test_known_presets(self):
        self.assertEqual(core.spec_from_preset("170x70").px, (2008, 827))
        self.assertEqual(core.spec_from_preset("85x35").px, (1004, 413))
        self.assertEqual(core.spec_from_preset("170X70", dpi=150).px, (1004, 413))

    def test_unknown_falls_back_to_parse(self):
        s = core.spec_from_preset("150x80", dpi=300)
        self.assertEqual((s.w_mm, s.h_mm), (150.0, 80.0))


class TestRotate(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((100, 200, 3), np.uint8)

    def test_90_swaps_dimensions(self):
        out = core.rotate90(self.img, 90)
        self.assertEqual(out.shape[:2], (200, 100))

    def test_180_keeps_dimensions(self):
        self.assertEqual(core.rotate90(self.img, 180).shape[:2], (100, 200))

    def test_270_and_zero(self):
        self.assertEqual(core.rotate90(self.img, 270).shape[:2], (200, 100))
        self.assertEqual(core.rotate90(self.img, 0).shape[:2], (100, 200))

    def test_invalid_angle_raises(self):
        with self.assertRaises(ValueError):
            core.rotate90(self.img, 45)


class TestAspectError(unittest.TestCase):
    def test_matching_ratio_is_small(self):
        q = np.float32([[0, 0], [970, 0], [970, 400], [0, 400]])  # 2.425 ~ 170/70
        self.assertLess(core.aspect_error(q, core.Spec(170, 70)), 0.02)

    def test_mismatched_ratio_is_flagged(self):
        q = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])  # 1.0 vs 2.43
        self.assertGreater(core.aspect_error(q, core.Spec(170, 70)), 0.5)


class TestBatchV02(unittest.TestCase):
    def _seed(self, d, n=2, broken=False):
        src = os.path.join(d, "in")
        os.makedirs(src)
        for i in range(n):
            img, _ = make_page()
            core.save(img, os.path.join(src, f"P-{i+1:03d}.png"), 300)
        if broken:
            open(os.path.join(src, "broken.png"), "wb").write(b"junk")
        return src

    def test_rotate_option_changes_output_size(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._seed(d, 1)
            dst = os.path.join(d, "out")
            cli.run_batch(src, dst, core.Spec(170, 70, 300), rotate=90)
            from PIL import Image
            im = Image.open(os.path.join(dst, os.listdir(dst)[0]))
            self.assertEqual(im.size, (827, 2008))

    def test_fail_dir_receives_copy(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._seed(d, 1, broken=True)
            dst, fail = os.path.join(d, "out"), os.path.join(d, "fail")
            ok, skip, _ = cli.run_batch(src, dst, core.Spec(170, 70, 300), fail_dir=fail)
            self.assertEqual((ok, skip), (1, 1))
            self.assertEqual(os.listdir(fail), ["broken.png"])

    def test_log_csv_written(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._seed(d, 1)
            dst = os.path.join(d, "out")
            log = os.path.join(d, "log.csv")
            cli.run_batch(src, dst, core.Spec(170, 70, 300), log_csv=log)
            with open(log, encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], cli.LOG_FIELDS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1][8], "성공")

    def test_no_overwrite_on_same_name(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._seed(d, 1)
            dst = os.path.join(d, "out")
            spec = core.Spec(170, 70, 300)
            cli.run_batch(src, dst, spec)
            first = os.listdir(dst)[0]
            cli.run_batch(src, dst, spec)
            files = sorted(os.listdir(dst))
            self.assertEqual(len(files), 2)
            stem, ext = os.path.splitext(first)
            self.assertEqual(files, sorted([first, f"{stem}_2{ext}"]))

    def test_tiff_output_keeps_dpi(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._seed(d, 1)
            dst = os.path.join(d, "out")
            cli.run_batch(src, dst, core.Spec(170, 70, 300), ext=".tif")
            from PIL import Image
            f = os.path.join(dst, os.listdir(dst)[0])
            self.assertTrue(f.endswith(".tif"))
            im = Image.open(f)
            self.assertEqual(im.size, (2008, 827))
            self.assertAlmostEqual(im.info.get("dpi", (0, 0))[0], 300, delta=1)


if __name__ == "__main__":
    unittest.main()
