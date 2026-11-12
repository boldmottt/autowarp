# AutoWarp

시편(스펙imen) 사진을 **실물 규격 그대로** 펴서 인쇄용 이미지로 뽑는 도구.

사진 한 장에 비스듬히 찍힌 시편의 네 귀퉁이를 잡으면, 지정한 실물 크기(mm)와 해상도(dpi)에 맞는 픽셀 크기로 원근을 보정해 저장한다. GUI 로 한 장씩, CLI 로 폴더 통째로 처리할 수 있다.

- 기본 규격 **170 × 70 mm @ 300 dpi = 2008 × 827 px**
- 자동 귀퉁이 검출 **기본 켜짐**
- 한 사진에 여러 개가 찍혀 있으면 각각 따로 뽑는다

## dewarp 과 무엇이 다른가

이 프로젝트는 [dewarp](https://github.com/bobm123/dewarp)(MIT, © 2025 Robert Marchese)의 코드를 기반으로 재작업했다. GUI 골격과 원근 변환·스케일 보정 로직은 그대로 쓰고, 실무에서 발목을 잡던 부분을 바꿨다.

| 항목 | dewarp | AutoWarp |
|---|---|---|
| 출력 픽셀 계산 | 170mm@300dpi → 2007px (버림) | **2008px (반올림, 인쇄 규격 일치)** |
| 여러 장 일괄 처리 | 없음 (한 장씩) | **`cli.py batch` 로 폴더 통째** |
| 여러 시편 자동 분리 | 없음 | 사진 1장에서 여러 개 검출 |
| 사진 테두리를 시편으로 오인 | 있음 | 프레임(면적 95% 초과)은 후보에서 제외 |
| 파일명 | `<원본>_dewarp.png` | `<원본>_170x70mm_2008x827px.png` (패턴 지정 가능) |
| dpi 메타데이터 | 저장 시 입력값 | 저장 시 규격 dpi 자동 기록 |
| 한글 경로 | `cv2.imread` 실패 | `imdecode` 로 처리 |
| 기본값 | 규격 없음 · 자동검출 꺼짐 | 시편 170×70 · 자동검출 켜짐 |

## 설치

파이썬 3.9 이상. GUI 는 tkinter 가 필요하다(macOS 기본 파이썬에는 Tk 8.5 만 있어 창이 안 뜰 수 있으니 Homebrew 파이썬 + `python-tk` 를 권장).

```bash
brew install python@3.12 python-tk@3.12          # macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## GUI

```bash
.venv/bin/python app.py                 # 빈 화면으로 시작
.venv/bin/python app.py 사진.jpg         # 바로 열기
```

1. 사진을 끌어다 놓으면 **자동으로 귀퉁이가 잡힌다** (안 맞으면 점을 끌어 조정)
2. 오른쪽에서 단위 `mm`, dpi `300`, 크기 `170 × 70` 입력 (프리셋 `시편 170x70` 선택 가능)
3. 크롭 모드에서 저장

## CLI (일괄)

```bash
# 폴더 통째로 → 규격 170x70mm, 300dpi
python cli.py batch ~/사진/시편 ~/사진/출력 --size 170x70 --dpi 300

# 파일명 패턴 지정 (시편_001_170x70mm.png)
python cli.py batch in out --size 170x70 --pattern "시편_{n:03d}_{w}x{h}mm"

# 미리보기만
python cli.py batch in out --size 170x70 --dry-run

# 한 장에 시편이 3개씩 찍혀 있으면
python cli.py batch in out --size 170x70 --per-page 3
```

출력 예시:

```
  ✔ M-01120_촬영.png  (1개 검출)
  ✔ G-60813_촬영.png  (1개 검출)

규격 170×70mm @ 300dpi = 2008×827px
완료 2장 · 건너뜀 0장 · 0.1초
```

읽기 실패·검출 실패 사진은 **건너뛰고 목록으로 보고**한다. 중간에 멈추지 않는다.

## 엔진 (`core.py`)

Tk 에 의존하지 않는 순수 함수라 다른 프로그램에서 그대로 쓸 수 있다.

```python
import core

spec = core.Spec(170, 70, 300)        # mm, mm, dpi
print(spec.px)                        # (2008, 827)

img = core.read_image("시편.jpg")
quad = core.best_quad(img, aspect=spec.aspect)   # 자동 검출 (실패하면 None)
out = core.warp(img, quad, spec)                 # 규격 픽셀로 변환
core.save(out, "결과.png", spec.dpi)              # dpi 메타데이터 포함
```

| 함수 | 역할 |
|---|---|
| `px_from_mm(mm, dpi)` | mm → px (반올림) |
| `Spec` | 출력 규격. `.px`, `.aspect`, `.label()`, `.parse("170x70")` |
| `detect_quads(img)` | 사각형 후보 전부 (큰 것부터) |
| `best_quad(img, aspect=)` | 비율까지 고려한 최적 후보 1개 |
| `warp(img, quad, spec)` | 원근 변환 |
| `save(img, path, dpi)` | PNG/JPEG/TIFF 로 dpi 포함 저장 |
| `output_name(stem, spec, ...)` | 파일명 규칙 |

## 테스트

```bash
python -m unittest discover -s tests      # 16개
```

합성 이미지(원근 왜곡된 흰 사각형)로 **검출 정확도·출력 픽셀·dpi 태그·한글 경로·배치 동작**을 검증한다.

## 구조

```
app.py          GUI (tkinter, dewarp 기반)
cli.py          일괄/단일 CLI
core.py         엔진 (검출·변환·저장·규격 계산)
lib/            dewarp 원본 라이브러리 (corner_detector, image_canvas, scale_calibrator, unit_converter)
tests/          unittest
```

## 라이선스

MIT. 원저작자 표기는 [LICENSE](LICENSE) 와 [NOTICE](NOTICE) 참고.
