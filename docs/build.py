#!/usr/bin/env python3
"""Build and combine the GeneralizedGradients.jl documentation.

Produces a single site in ``gh-pages/`` (at the repository root):

  * narrative docs (Sphinx + MyST + Furo)  -> gh-pages/      (site root)
  * API reference (Documenter.jl)          -> gh-pages/api/

Run from anywhere:  python docs/build.py
"""

import subprocess
import shutil
import sys
from pathlib import Path

docs_dir = Path(__file__).parent.resolve()
project_root = docs_dir.parent


def run(cmd, cwd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}  (in {cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


# 1. Instantiate the docs Julia environment (dev-link the package being documented).
print("==> Instantiating docs Julia environment…")
run(["julia", f"--project={docs_dir}", "-e",
     "using Pkg; Pkg.develop(PackageSpec(path = pwd())); Pkg.instantiate()"],
    cwd=project_root)

# 2. Build the API reference with Documenter (writes docs/api/build, incl. objects.inv
#    which Sphinx's intersphinx reads, so this must come before the Sphinx build).
print("\n==> Building API reference (Documenter.jl)…")
run(["julia", f"--project={docs_dir}", "docs/api/make.jl"], cwd=project_root)

# 3. Install the Sphinx toolchain.
print("\n==> Installing Sphinx dependencies…")
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=docs_dir)

# 4. Build the narrative docs with Sphinx.
print("\n==> Building narrative docs (Sphinx + Furo)…")
run(["sphinx-build", "-b", "html", "src", "build/html"], cwd=docs_dir)

# 5. Combine into gh-pages/  (Sphinx at root, Documenter under api/).
print("\n==> Combining into gh-pages/…")
gh_pages = project_root / "gh-pages"
if gh_pages.exists():
    shutil.rmtree(gh_pages)
gh_pages.mkdir()
shutil.copytree(docs_dir / "build" / "html", gh_pages, dirs_exist_ok=True)
shutil.copytree(docs_dir / "api" / "build", gh_pages / "api")
(gh_pages / ".nojekyll").touch()

print(f"\nDone! Combined site in {gh_pages}")
print(f"Open {gh_pages / 'index.html'} (API reference under api/index.html).")
