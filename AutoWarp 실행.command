#!/bin/bash
# 더블클릭 실행
cd "$(dirname "$0")" || exit 1
xattr -dr com.apple.quarantine .venv 2>/dev/null
xattr -d com.apple.quarantine "$0" 2>/dev/null
if [ ! -x .venv/bin/python ]; then
  echo "먼저 설치: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  read -n 1 -s; exit 1
fi
exec .venv/bin/python app.py "$@"
