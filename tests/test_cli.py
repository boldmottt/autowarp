"""CLI 배치 테스트 — 폴더 하나를 통째로 굴린다."""
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


class TestBatch(unittest.TestCase):
    def test_batch_writes_one_file_per_photo(self):
        spec = core.Spec(170, 70, 300)
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in"), os.path.join(d, "out")
            os.makedirs(src)
            for name in ("A-001", "B-002"):
                img, _ = make_page()
                core.save(img, os.path.join(src, name + ".png"), 300)
            ok, skip, notes = cli.run_batch(src, dst, spec)
            self.assertEqual((ok, skip), (2, 0))
            files = sorted(os.listdir(dst))
            self.assertEqual(len(files), 2)
            self.assertTrue(files[0].endswith("170x70mm_2008x827px.png"))

    def test_unreadable_photo_is_skipped_not_fatal(self):
        spec = core.Spec(170, 70, 300)
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in"), os.path.join(d, "out")
            os.makedirs(src)
            open(os.path.join(src, "broken.png"), "wb").write(b"not an image")
            img, _ = make_page()
            core.save(img, os.path.join(src, "good.png"), 300)
            ok, skip, notes = cli.run_batch(src, dst, spec)
            self.assertEqual(ok, 1)
            self.assertEqual(skip, 1)

    def test_dry_run_writes_nothing(self):
        spec = core.Spec(170, 70, 300)
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in"), os.path.join(d, "out")
            os.makedirs(src)
            img, _ = make_page()
            core.save(img, os.path.join(src, "x.png"), 300)
            ok, _, _ = cli.run_batch(src, dst, spec, dry=True)
            self.assertEqual(ok, 1)
            self.assertFalse(os.path.exists(dst) and os.listdir(dst))


if __name__ == "__main__":
    unittest.main()
