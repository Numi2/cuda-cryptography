#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT_DIR/build"}"

"$ROOT_DIR/scripts/build.sh" -DCMAKE_BUILD_TYPE=Release "$@"
"$BUILD_DIR/cpb_bench"
