"""autowarp 엔진 테스트 — 합성 이미지로 규격·검출·변환을 검증한다."""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402


def make_page(w=1600, h=1200, quad=None, bg=30):
    """어두운 배경 + 흰 사각형(원근 왜곡) 이미지. 흰 시편이 quad 자리에 놓인다."""
    img = np.full((h, w, 3), bg, np.uint8)
    if quad is None:
        quad = np.float32([[300, 250], [1250, 300], [1200, 950], [340, 900]])
    flat = np.float32([[0, 0], [999, 0], [999, 499], [0, 499]])  # 1000x500 평면
    m = cv2.getPerspectiveTransform(flat, np.float32(quad))
    plate = np.full((500, 1000, 3), 235, np.uint8)
    warped = cv2.warpPerspective(
        plate, m, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(bg, bg, bg),
    )
    mask = warped.max(axis=2) > bg + 20
    img[mask] = warped[mask]
    return img, np.float32(quad)


class TestUnits(unittest.TestCase):
    def test_170x70_300dpi_is_exact_print_pixels(self):
        s = core.Spec(170, 70, 300)
        self.assertEqual(s.px, (2008, 827))

    def test_other_dpi(self):
        self.assertEqual(core.Spec(170, 70, 150).px, (1004, 413))
        self.assertEqual(core.Spec(210, 297, 300).px, (2480, 3508))  # A4

    def test_spec_parse(self):
        self.assertEqual(core.Spec.parse("170x70").w_mm, 170.0)
        self.assertEqual(core.Spec.parse("170×70", dpi=150).dpi, 150.0)
        with self.assertRaises(ValueError):
            core.Spec.parse("170")

    def test_mm_px_roundtrip(self):
        self.assertAlmostEqual(core.mm_from_px(core.px_from_mm(170, 300), 300), 170.0, delta=0.05)


class TestOrdering(unittest.TestCase):
    def test_quad_ordering_any_input_order(self):
        pts = [[1200, 950], [300, 250], [340, 900], [1250, 300]]
        q = core.order_quad(pts)
        self.assertLess(q[0][0], q[1][0])        # tl.x < tr.x
        self.assertLess(q[0][1], q[3][1])        # tl.y < bl.y
        self.assertGreater(q[2][1], q[1][1])     # br.y > tr.y


class TestDetect(unittest.TestCase):
    def test_finds_the_specimen(self):
        img, gt = make_page()
        quads = core.detect_quads(img)
        self.assertTrue(quads, "사각형을 못 찾음")
        got = quads[0]
        for p in gt:
            d = np.min(np.linalg.norm(got - p, axis=1))
            self.assertLess(d, 40, f"꼭짓점 오차 큼: {d:.1f}px")

    def test_blank_image_returns_nothing(self):
        img = np.full((600, 800, 3), 128, np.uint8)
        self.assertEqual(core.detect_quads(img), [])

    def test_multiple_specimens(self):
        img = np.full((900, 1400, 3), 25, np.uint8)
        cv2.rectangle(img, (80, 120), (560, 420), (240, 240, 240), -1)
        cv2.rectangle(img, (700, 180), (1300, 630), (235, 235, 235), -1)
        quads = core.detect_quads(img)
        self.assertGreaterEqual(len(quads), 2)


class TestWarp(unittest.TestCase):
    def test_output_is_exact_spec_pixels(self):
        img, gt = make_page()
        q = core.best_quad(img, aspect=170 / 70)
        self.assertIsNotNone(q)
        out = core.warp(img, q, core.Spec(170, 70, 300))
        self.assertEqual((out.shape[1], out.shape[0]), (2008, 827))

    def test_warp_fills_with_white(self):
        img, gt = make_page()
        w = core.warp(img, gt, core.Spec(170, 70, 150))
        inner = w[20:-20, 20:-20]
        self.assertGreater(float(inner.mean()), 200, "안쪽이 시편(흰색)이 아님")

    def test_keep_ratio(self):
        """keep_ratio 는 규격 상자 안에서 원본 비율을 유지한다(안 늘어남)."""
        img, gt = make_page()
        q = core.order_quad(gt)
        src_aspect = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2 / (
            (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2
        )
        out = core.warp(img, gt, core.Spec(170, 70, 300), keep_ratio=True)
        got = out.shape[1] / out.shape[0]
        self.assertAlmostEqual(got, src_aspect, delta=0.05)
        self.assertLessEqual(out.shape[1], 2008)
        self.assertLessEqual(out.shape[0], 827)

    def test_auto_warp_and_save(self):
        img, _ = make_page()
        out = core.auto_warp(img, core.Spec(170, 70, 300))
        self.assertIsNotNone(out)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, core.output_name("M-01120", core.Spec(170, 70, 300)))
            core.save(out, p, 300)
            self.assertTrue(os.path.exists(p))
            from PIL import Image
            im = Image.open(p)
            self.assertEqual(im.size, (2008, 827))
            dpi = im.info.get("dpi", (0, 0))[0]
            self.assertAlmostEqual(dpi, 300, delta=1)

    def test_korean_path(self):
        img, _ = make_page()
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "시편 사진")
            os.makedirs(sub)
            src = os.path.join(sub, "테스트.png")
            core.save(img, src, 300)
            self.assertIsNotNone(core.read_image(src))
            self.assertEqual(len(core.iter_images(sub)), 1)


if __name__ == "__main__":
    unittest.main()
