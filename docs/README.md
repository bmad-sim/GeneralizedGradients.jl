# Documentation

The documentation site is built from **two engines** and combined into one site:

- **MyST** ([mystmd](https://mystmd.org)) renders the narrative/general docs in
  `docs/myst/` — this becomes the site root.
- **Documenter.jl** renders the API reference from the package docstrings
  (`docs/make.jl`, `docs/src/`) — this becomes the `/api/` sub-site.

`.github/workflows/docs.yml` builds both, assembles them
(`site/` = MyST at root + Documenter at `site/api/`), and publishes to the
`gh-pages` branch. Pull requests get a full preview at
`previews/PR<number>/` with a link posted as a PR comment; the preview is
deleted on PR close by `.github/workflows/docs-cleanup.yml`.

## One-time repository setup

1. In **Settings → Pages**, set the source to **Deploy from a branch**, branch
   **`gh-pages`**, folder **`/ (root)`**. (The first push to `main` creates the
   branch.)
2. Ensure **Settings → Actions → General → Workflow permissions** is set to
   **Read and write permissions** so the workflow can push to `gh-pages` and
   comment on PRs.

The published site will be at
<https://bmad-sim.github.io/GeneralizedGradients.jl/>.

> **Note on fork PRs:** previews are deployed by pushing to `gh-pages`. Pull
> requests opened from a *fork* have a read-only token and cannot deploy a
> preview; PRs from branches within this repository work normally.

## Building locally

### Narrative docs (MyST)

Requires [Node.js](https://nodejs.org) and `mystmd`:

```sh
npm install -g mystmd
cd docs/myst
myst start          # live-reloading preview at http://localhost:3000
# or a static build:
myst build --html   # output in docs/myst/_build/html/
```

### API reference (Documenter)

Requires Julia:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
# output in docs/build/ (open docs/build/index.html)
```

### Combined site

To reproduce the deployed layout locally:

```sh
rm -rf site && mkdir -p site/api
cp -r docs/myst/_build/html/. site/
cp -r docs/build/. site/api/
```
