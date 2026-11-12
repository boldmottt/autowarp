"""autowarp 엔진 — 시편 사진의 원근을 펴고, 실물 규격(mm/dpi)에 맞는 픽셀로 저장한다.

GUI(app.py)와 CLI(cli.py)가 같이 쓰는 순수 함수 모음. Tk 의존 없음.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

MM_PER_INCH = 25.4
JPEG_EXT = {".jpg", ".jpeg"}
RAW_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}


# ---------------------------------------------------------------- 규격 계산
def px_from_mm(mm: float, dpi: float) -> int:
    """mm → 픽셀. 반올림(0.5 올림)한다. 170mm@300dpi = 2008px."""
    if dpi <= 0:
        raise ValueError("dpi 는 0보다 커야 합니다")
    return int(math.floor(mm / MM_PER_INCH * dpi + 0.5))


def mm_from_px(px: float, dpi: float) -> float:
    if dpi <= 0:
        raise ValueError("dpi 는 0보다 커야 합니다")
    return px / dpi * MM_PER_INCH


@dataclass(frozen=True)
class Spec:
    """출력 규격: 실물 크기(mm) + 해상도(dpi)."""

    w_mm: float = 170.0
    h_mm: float = 70.0
    dpi: float = 300.0

    @property
    def px(self) -> tuple[int, int]:
        return px_from_mm(self.w_mm, self.dpi), px_from_mm(self.h_mm, self.dpi)

    @property
    def aspect(self) -> float:
        return self.w_mm / self.h_mm

    def label(self) -> str:
        w, h = self.px
        return f"{self.w_mm:g}x{self.h_mm:g}mm_{w}x{h}px_{self.dpi:g}dpi"

    @classmethod
    def parse(cls, text: str, dpi: float = 300.0) -> "Spec":
        """'170x70' / '170X70' / '170,70' 형태를 읽는다."""
        t = text.lower().replace("×", "x").replace(",", "x").replace(" ", "")
        if "x" not in t:
            raise ValueError(f"규격 형식이 아닙니다: {text!r} (예: 170x70)")
        a, b = t.split("x", 1)
        return cls(float(a), float(b), float(dpi))


# ---------------------------------------------------------------- 사각형 정리
def _dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def order_quad(pts: Sequence[Sequence[float]]) -> np.ndarray:
    """네 점을 좌상 → 우상 → 우하 → 좌하 순서로 정렬한다(회전에 견고)."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()
    tl = p[int(np.argmin(s))]
    br = p[int(np.argmax(s))]
    tr = p[int(np.argmin(d))]
    bl = p[int(np.argmax(d))]
    quad = np.array([tl, tr, br, bl], dtype=np.float64)
    # 극단값이 겹치면(마름모꼴) 무게중심 각도 정렬로 폴백
    if len({tuple(q) for q in quad}) < 4:
        c = p.mean(axis=0)
        ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
        p = p[np.argsort(ang)]
        quad = np.array([p[0], p[1], p[2], p[3]], dtype=np.float64)
    return quad


def quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)))


def _area_ratio(quad: np.ndarray, shape) -> float:
    h, w = shape[:2]
    return quad_area(quad) / float(w * h) if w * h else 0.0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ay0, ay1 = float(np.min(a[:, 1])), float(np.max(a[:, 1]))
    ax0, ax1 = float(np.min(a[:, 0])), float(np.max(a[:, 0]))
    by0, by1 = float(np.min(b[:, 1])), float(np.max(b[:, 1]))
    bx0, bx1 = float(np.min(b[:, 0])), float(np.max(b[:, 0]))
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- 자동 검출
def detect_quads(
    bgr: np.ndarray,
    min_area_ratio: float = 0.03,
    max_area_ratio: float = 0.95,
    max_quads: int = 12,
    max_iou: float = 0.60,
) -> list[np.ndarray]:
    """사진 안의 사각형(시편) 후보를 큰 것부터 돌려준다.

    여러 장이 한 컷에 찍힌 경우를 위해 후보를 여러 개 낸다.
    """
    if bgr is None or bgr.size == 0:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    masks = []
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, m = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
        masks.append(m)
    try:
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        masks.append(edges)
    except cv2.error:
        pass

    found: list[np.ndarray] = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        for mode in (cv2.RETR_EXTERNAL, cv2.RETR_LIST):
            contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
                if _area_ratio(np.array(cv2.boxPoints(cv2.minAreaRect(c))), bgr.shape) < min_area_ratio:
                    continue  # 너무 작은 조각
                peri = cv2.arcLength(c, True)
                for eps in (0.02, 0.04, 0.06):
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    if len(approx) != 4 or not cv2.isContourConvex(approx):
                        continue
                    q = order_quad(approx.reshape(4, 2))
                    ar = _area_ratio(q, bgr.shape)
                    if quad_area(q) <= 0 or ar < min_area_ratio:
                        break
                    if ar > max_area_ratio:
                        break  # 사진 테두리 전체 = 시편이 아니라 배경 프레임
                    found.append(q)
                    break

    found.sort(key=quad_area, reverse=True)
    out: list[np.ndarray] = []
    for q in found:
        if all(_iou(q, k) < max_iou for k in out):
            out.append(q)
        if len(out) >= max_quads:
            break
    return out


def best_quad(bgr: np.ndarray, aspect: float | None = None, min_area_ratio: float = 0.03):
    """가장 그럴듯한 사각형 하나. aspect 를 주면 비율이 가까운 것을 고른다."""
    quads = detect_quads(bgr, min_area_ratio=min_area_ratio)
    if not quads:
        return None
    if aspect is None:
        return quads[0]

    def score(q):
        w = (_dist(q[0], q[1]) + _dist(q[3], q[2])) / 2
        h = (_dist(q[0], q[3]) + _dist(q[1], q[2])) / 2
        if min(w, h) <= 0:
            return 99.0
        return abs(math.log((w / h) / aspect)) - 0.02 * math.log(quad_area(q))

    return min(quads, key=score)


# ---------------------------------------------------------------- 변환
def warp(
    bgr: np.ndarray,
    quad: Sequence[Sequence[float]],
    spec: Spec,
    keep_ratio: bool = False,
    interp: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """사각형 영역을 규격 픽셀 크기로 편다."""
    w_px, h_px = spec.px
    if keep_ratio:
        q = order_quad(quad)
        w = (_dist(q[0], q[1]) + _dist(q[3], q[2])) / 2
        h = (_dist(q[0], q[3]) + _dist(q[1], q[2])) / 2
        if w > 0 and h > 0:
            if w / h >= spec.aspect:
                h_px = max(1, int(round(w_px / (w / h))))
            else:
                w_px = max(1, int(round(h_px * (w / h))))

    src = order_quad(quad).astype(np.float32)
    dst = np.array(
        [[0, 0], [w_px - 1, 0], [w_px - 1, h_px - 1], [0, h_px - 1]], dtype=np.float32
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        bgr, m, (w_px, h_px), flags=interp, borderMode=cv2.BORDER_REPLICATE
    )


def auto_warp(bgr: np.ndarray, spec: Spec, **kw) -> np.ndarray | None:
    """자동 검출 후 바로 편다. 검출 실패면 None."""
    q = best_quad(bgr, aspect=spec.aspect if spec else None)
    if q is None:
        return None
    return warp(bgr, q, spec, **kw)


# ---------------------------------------------------------------- 저장
def save(img_bgr: np.ndarray, path: str, dpi: float = 300.0, quality: int = 95) -> str:
    """dpi 메타데이터를 넣어 저장한다(PNG 는 pHYs, JPEG/TIFF 는 dpi 태그)."""
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else img_bgr
    im = Image.fromarray(rgb)
    ext = os.path.splitext(path)[1].lower()
    kw = {"dpi": (float(dpi), float(dpi))}
    if ext in JPEG_EXT:
        kw.update(quality=quality, subsampling=0, optimize=True)
    elif ext in (".tif", ".tiff"):
        kw.update(compression="tiff_lzw")
    im.save(path, **kw)
    return path


def output_name(stem: str, spec: Spec, index: int | None = None, pattern: str | None = None) -> str:
    """기본 파일명: <원본>_<w>x<h>mm_<px>x<px>px.png"""
    w_px, h_px = spec.px
    if pattern:
        return pattern.format(
            stem=stem,
            n=index if index is not None else 1,
            w=spec.w_mm,
            h=spec.h_mm,
            dpi=spec.dpi,
            wpx=w_px,
            hpx=h_px,
        )
    return f"{stem}_{spec.w_mm:g}x{spec.h_mm:g}mm_{w_px}x{h_px}px.png"


def read_image(path: str) -> np.ndarray | None:
    """한글 경로에서도 읽는다(cv2.imread 는 유니코드 경로에 약함)."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def iter_images(folder: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("."):
            continue
        if os.path.splitext(name)[1].lower() in RAW_EXT:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                out.append(p)
    return out

# ---------------------------------------------------------------- 프리셋
PRESETS: dict[str, tuple[float, float]] = {
    "170x70": (170.0, 70.0),
    "85x35": (85.0, 35.0),
    "210x297": (210.0, 297.0),
    "297x210": (297.0, 210.0),
}


def spec_from_preset(name: str, dpi: float = 300.0) -> "Spec":
    """프리셋 이름('170x70') 또는 직접 입력('150x80')을 Spec 으로."""
    key = name.lower().replace("×", "x").replace(" ", "")
    if key in PRESETS:
        w, h = PRESETS[key]
        return Spec(w, h, dpi)
    return Spec.parse(name, dpi)


# ---------------------------------------------------------------- 회전
def rotate90(img: np.ndarray, degrees: int) -> np.ndarray:
    """90도 단위 회전. 0/90/180/270 만 허용."""
    d = int(degrees) % 360
    if d == 0:
        return img
    if d == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"회전은 0/90/180/270 만: {degrees}")


# ---------------------------------------------------------------- 비율 진단
def aspect_error(quad: Sequence[Sequence[float]], spec: "Spec") -> float:
    """검출된 사각형의 가로/세로 비율이 규격과 얼마나 다른지(상대값).

    0 이면 일치. 0.1 이면 10% 늘어나거나 줄어든다는 뜻.
    """
    q = order_quad(quad)
    w = (_dist(q[0], q[1]) + _dist(q[3], q[2])) / 2
    h = (_dist(q[0], q[3]) + _dist(q[1], q[2])) / 2
    if w <= 0 or h <= 0:
        return float("inf")
    return abs((w / h) / spec.aspect - 1.0)
