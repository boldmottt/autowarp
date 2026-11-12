#!/usr/bin/env python3
"""autowarp CLI — 폴더 단위로 시편 사진을 한 번에 편다.

    python cli.py batch <입력폴더> <출력폴더> [--preset 170x70] [--dpi 300]
    python cli.py one   <사진> [--preset 170x70]

검출에 실패한 사진은 실패 폴더로 복사하고 목록으로 보고한다(중간에 멈추지 않는다).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime

import core

LOG_FIELDS = ["시각", "원본", "출력", "규격", "dpi", "픽셀", "회전", "비율차이", "결과", "비고"]


def _spec(args) -> core.Spec:
    if getattr(args, "preset", None):
        return core.spec_from_preset(args.preset, args.dpi)
    if getattr(args, "size", None):
        return core.Spec.parse(args.size, args.dpi)
    return core.Spec(dpi=args.dpi)


def _unique(path: str) -> str:
    """같은 이름이 있으면 _2, _3 … 을 붙인다(덮어쓰기 금지)."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{stem}_{i}{ext}"):
        i += 1
    return f"{stem}_{i}{ext}"


def run_batch(
    src,
    dst,
    spec,
    per_page=1,
    dry=False,
    pattern=None,
    ext=".png",
    rotate=0,
    fail_dir=None,
    log_csv=None,
    warn_ratio=0.10,
):
    files = core.iter_images(src)
    if not files:
        print(f"이미지 없음: {src}")
        return 0, 0, []
    os.makedirs(dst, exist_ok=True)
    if fail_dir and not dry:
        os.makedirs(fail_dir, exist_ok=True)

    rows, notes = [], []
    ok = skip = 0
    t0 = time.time()
    w_px, h_px = spec.px

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = core.read_image(path)
        if img is None:
            notes.append(f"읽기 실패: {stem}")
            skip += 1
            if fail_dir and not dry:
                shutil.copy2(path, _unique(os.path.join(fail_dir, os.path.basename(path))))
            rows.append([datetime.now().isoformat(timespec="seconds"), path, "", spec.label(), spec.dpi,
                         f"{w_px}x{h_px}", rotate, "", "실패", "읽기 실패"])
            continue

        quads = core.detect_quads(img)
        if not quads:
            notes.append(f"사각형 못 찾음: {stem}")
            skip += 1
            if fail_dir and not dry:
                shutil.copy2(path, _unique(os.path.join(fail_dir, os.path.basename(path))))
            rows.append([datetime.now().isoformat(timespec="seconds"), path, "", spec.label(), spec.dpi,
                         f"{w_px}x{h_px}", rotate, "", "실패", "검출 실패"])
            continue

        quads = quads[: max(1, per_page)]
        for qi, q in enumerate(quads, start=1):
            idx = qi if len(quads) > 1 else None
            name = core.output_name(stem, spec, idx, pattern)
            name = os.path.splitext(name)[0] + ext
            out = _unique(os.path.join(dst, name))
            err = core.aspect_error(q, spec)
            flag = "" if err <= warn_ratio else f"비율 {err*100:.0f}% 차이"
            if dry:
                notes.append(f"[예정] {os.path.basename(path)} → {os.path.basename(out)}")
                ok += 1
                continue
            warped = core.warp(img, q, spec)
            if rotate:
                warped = core.rotate90(warped, rotate)
            core.save(warped, out, spec.dpi)
            rows.append([datetime.now().isoformat(timespec="seconds"), path, out, spec.label(), spec.dpi,
                         f"{warped.shape[1]}x{warped.shape[0]}", rotate,
                         f"{err*100:.1f}%", "성공", flag])
            ok += 1
        mark = "·" if dry else ("!" if flag else "✔")
        print(f"  {mark} {os.path.basename(path)}  ({len(quads)}개 검출){'  ' + flag if flag else ''}")

    if log_csv and rows and not dry:
        with open(log_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(LOG_FIELDS)
            w.writerows(rows)

    dt = time.time() - t0
    print()
    print(f"규격 {spec.w_mm:g}×{spec.h_mm:g}mm @ {spec.dpi:g}dpi = {w_px}×{h_px}px")
    print(f"완료 {ok}장 · 건너뜀 {skip}장 · {dt:.1f}초")
    if ok and not dry:
        print(f"장당 평균 {dt/max(ok+skip,1):.2f}초")
    for n in notes:
        print("  -", n)
    return ok, skip, notes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autowarp", description="시편 사진 원근 보정")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        g = p.add_mutually_exclusive_group()
        g.add_argument("--preset", help="규격 프리셋 (예: 170x70, 85x35, 297x210)")
        g.add_argument("--size", help="직접 지정 (예: 150x80)")
        p.add_argument("--dpi", type=float, default=300.0, help="해상도 (기본 300)")
        p.add_argument("--ext", default=".png", help="출력 확장자 .png/.jpg/.tif")
        p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                       help="결과를 90도 단위로 돌린다")

    b = sub.add_parser("batch", help="폴더 통째로")
    b.add_argument("src")
    b.add_argument("dst")
    b.add_argument("--per-page", type=int, default=1, help="한 사진에서 뽑을 시편 수 (기본 1)")
    b.add_argument("--pattern", help="파일명 패턴 (예: '시편_{n:03d}_{w}x{h}mm')")
    b.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 예정만 출력")
    b.add_argument("--fail-dir", help="검출 실패 사진을 복사할 폴더")
    b.add_argument("--log", dest="log_csv", help="처리 이력 CSV 경로")
    b.add_argument("--warn-ratio", type=float, default=0.10,
                   help="규격 비율과 이만큼(비율) 이상 다르면 경고 (기본 0.10)")
    common(b)

    o = sub.add_parser("one", help="사진 한 장")
    o.add_argument("src")
    o.add_argument("--out", default=None, help="출력 폴더 (기본: 원본 옆)")
    common(o)

    a = ap.parse_args(argv)
    spec = _spec(a)

    if a.cmd == "batch":
        ok, skip, _ = run_batch(a.src, a.dst, spec, a.per_page, a.dry_run, a.pattern, a.ext,
                                a.rotate, a.fail_dir, a.log_csv, a.warn_ratio)
        return 0 if ok else 1

    src = os.path.abspath(a.src)
    out_dir = a.out or os.path.dirname(src)
    ok, skip, _ = run_batch(os.path.dirname(src), out_dir, spec, 1, False, None, a.ext,
                            a.rotate, None, None, 0.10)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
