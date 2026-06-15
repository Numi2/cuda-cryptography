#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
CHECK=0

if [[ "${1:-}" == "--check" ]]; then
  CHECK=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

if ! command -v "$CLANG_FORMAT" >/dev/null 2>&1; then
  echo "clang-format not found. Install clang-format or set CLANG_FORMAT=/path/to/clang-format." >&2
  exit 1
fi

if [[ "$CHECK" -eq 1 ]]; then
  find "$ROOT_DIR/include" "$ROOT_DIR/src" "$ROOT_DIR/tests" "$ROOT_DIR/bench" \
    -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) -print0 |
    xargs -0 "$CLANG_FORMAT" --dry-run --Werror
else
  find "$ROOT_DIR/include" "$ROOT_DIR/src" "$ROOT_DIR/tests" "$ROOT_DIR/bench" \
    -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) -print0 |
    xargs -0 "$CLANG_FORMAT" -i
fi
