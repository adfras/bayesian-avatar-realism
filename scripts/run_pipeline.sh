#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export PKG_CONFIG_PATH="${ROOT_DIR}/.deps/root/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
export C_INCLUDE_PATH="${ROOT_DIR}/.deps/root/usr/include:${ROOT_DIR}/.deps/root/usr/include/freetype2:${ROOT_DIR}/.deps/root/usr/include/harfbuzz:${ROOT_DIR}/.deps/root/usr/include/fribidi:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${C_INCLUDE_PATH}"
export LIBRARY_PATH="${ROOT_DIR}/.deps/root/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ROOT_DIR}/.deps/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="${ROOT_DIR}/.pyhome/bin:${PATH}"
export R_LIBS_USER="${ROOT_DIR}/R/library"

exec Rscript "${ROOT_DIR}/R/run_pipeline.R" "$@"
