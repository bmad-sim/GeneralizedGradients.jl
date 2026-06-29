# Documentation

The documentation site is built from **two engines** and combined into one site,
mirroring the [SciBmad.jl](https://bmad-sim.github.io/SciBmad.jl/) docs:

- **Sphinx + MyST + Furo** renders the narrative/general docs in `docs/src/`
  (MyST Markdown, `conf.py`) — this becomes the site root.
- **Documenter.jl** renders the API reference from the package docstrings
  (`docs/api/make.jl`, `docs/api/src/`) — this becomes the `/api/` sub-site.

`docs/build.py` builds both and combines them into `gh-pages/` (Sphinx at the
root, Documenter under `gh-pages/api/`). The unified look comes from the Furo
theme plus an **"API Reference →"** link in the Furo sidebar
(`docs/src/_templates/sidebar-external-links.html`) and a **"← Documentation"**
back-link in the Documenter site (`docs/api/src/main-docs.md`).

`.github/workflows/docs.yml` runs `docs/build.py` and publishes `gh-pages/` to
the `gh-pages` branch. Pull requests get a full preview at
`previews/PR<number>/` with a link posted as a PR comment; the preview is
deleted on PR close by `.github/workflows/docs-cleanup.yml`.

## One-time repository setup

1. In **Settings → Pages**, set the source to **Deploy from a branch**, branch
   **`gh-pages`**, folder **`/ (root)`**.
2. Ensure **Settings → Actions → General → Workflow permissions** is set to
   **Read and write permissions** so the workflow can push to `gh-pages` and
   comment on PRs.

The published site is at <https://bmad-sim.github.io/GeneralizedGradients.jl/>.

> **Note on fork PRs:** previews deploy by pushing to `gh-pages`; PRs opened from
> a *fork* have a read-only token and cannot deploy a preview. PRs from branches
> within this repository work normally.

## Viewing the documentation locally

The easiest way is the helper script [`docs/build_local.sh`](build_local.sh),
which builds the combined site and serves it:

```sh
docs/build_local.sh
```

Then open:

- Narrative docs: <http://localhost:8000/>
- API reference: <http://localhost:8000/api/>

Press `Ctrl-C` to stop. Options: `--port 9000`, `--no-serve`. Requirements:
`julia` and `python3` (the Sphinx toolchain is pip-installed automatically from
`requirements.txt`).

## Building manually

```sh
python docs/build.py        # builds both engines -> gh-pages/
```

To work on just one half:

```sh
# API reference (Documenter):
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/api/make.jl          # -> docs/api/build/

# Narrative docs (Sphinx): from docs/, after pip-installing requirements.txt
cd docs && sphinx-build -b html src build/html  # -> docs/build/html/
```
