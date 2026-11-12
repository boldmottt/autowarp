#!/usr/bin/env python3
"""autowarp CLI — 폴더 단위로 시편 사진을 한 번에 편다.

    python cli.py batch <입력폴더> <출력폴더> [--size 170x70] [--dpi 300]
    python cli.py one   <사진> [--size 170x70] [--dpi 300] [--out 폴더]

검출에 실패한 사진은 건너뛰고 목록으로 보고한다(중간에 멈추지 않는다).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import core


def _spec(args) -> core.Spec:
    if args.size:
        w, h = args.size.lower().replace("×", "x").split("x")
        return core.Spec(float(w), float(h), args.dpi)
    return core.Spec(dpi=args.dpi)


def run_batch(src, dst, spec, quads_per_page=1, dry=False, pattern=None, ext=".png"):
    files = core.iter_images(src)
    if not files:
        print(f"이미지 없음: {src}")
        return 0, 0, []
    os.makedirs(dst, exist_ok=True)
    ok, skip, notes = 0, 0, []
    t0 = time.time()
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = core.read_image(path)
        if img is None:
            notes.append(f"읽기 실패: {stem}")
            skip += 1
            continue
        quads = core.detect_quads(img)
        if not quads:
            notes.append(f"사각형 못 찾음: {stem}")
            skip += 1
            continue
        quads = quads[: max(1, quads_per_page)]
        for qi, q in enumerate(quads, start=1):
            idx = qi if len(quads) > 1 else None
            name = core.output_name(stem, spec, idx, pattern)
            name = os.path.splitext(name)[0] + ext
            out = os.path.join(dst, name)
            if dry:
                notes.append(f"[예정] {os.path.basename(path)} → {name}")
                ok += 1
                continue
            warped = core.warp(img, q, spec)
            core.save(warped, out, spec.dpi)
            ok += 1
        print(f"  {'·' if dry else '✔'} {os.path.basename(path)}  ({len(quads)}개 검출)")
    dt = time.time() - t0
    w_px, h_px = spec.px
    print()
    print(f"규격 {spec.w_mm:g}×{spec.h_mm:g}mm @ {spec.dpi:g}dpi = {w_px}×{h_px}px")
    print(f"완료 {ok}장 · 건너뜀 {skip}장 · {dt:.1f}초")
    for n in notes:
        print("  -", n)
    return ok, skip, notes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autowarp", description="시편 사진 원근 보정")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--size", help="출력 실물 크기 mm (예: 170x70)")
        p.add_argument("--dpi", type=float, default=300.0, help="해상도 (기본 300)")
        p.add_argument("--ext", default=".png", help="출력 확장자 (.png/.jpg/.tif)")

    b = sub.add_parser("batch", help="폴더 통째로")
    b.add_argument("src")
    b.add_argument("dst")
    b.add_argument("--per-page", type=int, default=1, help="한 사진에서 뽑을 시편 수 (기본 1)")
    b.add_argument("--pattern", help="파일명 패턴 (예: '시편_{n:03d}_{w}x{h}mm')")
    b.add_argument("--dry-run", action="store_true")
    common(b)

    o = sub.add_parser("one", help="사진 한 장")
    o.add_argument("src")
    o.add_argument("--out", default=None, help="출력 폴더 (기본: 원본 옆)")
    common(o)

    a = ap.parse_args(argv)
    spec = _spec(a)
    if a.cmd == "batch":
        ok, skip, _ = run_batch(a.src, a.dst, spec, a.per_page, a.dry_run, a.pattern, a.ext)
        return 0 if ok else 1

    out_dir = a.out or os.path.dirname(os.path.abspath(a.src))
    ok, skip, _ = run_batch(os.path.dirname(os.path.abspath(a.src)), out_dir, spec, 1, False, None, a.ext)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
