#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# docs/build_local.sh
#
# Build the full documentation site locally and serve it for viewing.
# Reproduces what .github/workflows/docs.yml does on CI:
#   * Documenter -> docs/build/        (API reference, served at /api/)
#   * MyST       -> docs/myst/_build/html/   (narrative docs, served at root)
#   * combined   -> site/              (MyST at root + Documenter under api/)
# and then starts a local web server so links between the two engines work.
#
# Usage:
#   docs/build_local.sh                 # build, then serve at http://localhost:8000/
#   docs/build_local.sh --port 9000     # serve on a different port
#   docs/build_local.sh --no-serve      # just build site/, don't start a server
#
# Requirements: julia, and mystmd (`npm install -g mystmd`). Serving uses
# python3 if available, otherwise `npx serve`.
# ---------------------------------------------------------------------------
set -euo pipefail

PORT=8000
SERVE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --no-serve) SERVE=0 ;;
    --port) PORT="$2"; shift ;;
    --port=*) PORT="${1#*=}" ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# Repo root = parent of this script's directory (works from any CWD).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

command -v julia >/dev/null || { echo "ERROR: 'julia' not found in PATH." >&2; exit 1; }
command -v myst  >/dev/null || { echo "ERROR: 'myst' not found. Install with: npm install -g mystmd" >&2; exit 1; }

echo "==> [1/4] Building API reference (Documenter)…"
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path = pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl

echo "==> [2/4] Building narrative docs (MyST)…"
# Empty BASE_URL => assets/links resolve from the server root (localhost).
( cd docs/myst && BASE_URL="" myst build --html )

echo "==> [3/4] Assembling combined site/…"
rm -rf site
mkdir -p site/api
cp -r docs/myst/_build/html/. site/
cp -r docs/build/. site/api/
touch site/.nojekyll

echo "==> [4/4] Done. Combined site is in: $ROOT/site"

if [ "$SERVE" -eq 0 ]; then
  echo "    Open site/index.html, or serve it with: python3 -m http.server --directory site"
  exit 0
fi

echo
echo "    Narrative docs : http://localhost:$PORT/"
echo "    API reference  : http://localhost:$PORT/api/"
echo "    (Press Ctrl-C to stop the server.)"
echo

if command -v python3 >/dev/null; then
  exec python3 -m http.server "$PORT" --directory site
elif command -v npx >/dev/null; then
  exec npx --yes serve -l "$PORT" site
else
  echo "ERROR: need python3 or npx to serve. Combined site is ready in site/." >&2
  exit 1
fi
