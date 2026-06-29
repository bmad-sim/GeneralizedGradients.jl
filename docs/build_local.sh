#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# docs/build_local.sh
#
# Build the full documentation site locally and serve it for viewing.
# Runs docs/build.py (Documenter API + Sphinx/Furo narrative -> ./gh-pages),
# then starts a local web server so the links between the narrative docs and the
# /api/ reference work.
#
# Usage:
#   docs/build_local.sh                 # build, then serve at http://localhost:8000/
#   docs/build_local.sh --port 9000     # serve on a different port
#   docs/build_local.sh --no-serve      # just build gh-pages/, don't start a server
#
# Requirements: julia and python3 (the Sphinx toolchain is pip-installed by
# docs/build.py from docs/requirements.txt).
# ---------------------------------------------------------------------------
set -euo pipefail

PORT=8000
SERVE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --no-serve) SERVE=0 ;;
    --port) PORT="$2"; shift ;;
    --port=*) PORT="${1#*=}" ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

command -v julia   >/dev/null || { echo "ERROR: 'julia' not found in PATH." >&2; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: 'python3' not found in PATH." >&2; exit 1; }

echo "==> Building combined documentation (docs/build.py)…"
python3 docs/build.py

echo "==> Done. Combined site is in: $ROOT/gh-pages"
if [ "$SERVE" -eq 0 ]; then
  echo "    Open gh-pages/index.html, or serve with: python3 -m http.server --directory gh-pages"
  exit 0
fi

echo
echo "    Narrative docs : http://localhost:$PORT/"
echo "    API reference  : http://localhost:$PORT/api/"
echo "    (Press Ctrl-C to stop the server.)"
echo
exec python3 -m http.server "$PORT" --directory gh-pages
