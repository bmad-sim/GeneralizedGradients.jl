# ---------------------------------------------------------------------------
# docs/make.jl
#
# Build the API reference (from the package docstrings) with Documenter.jl.
# This produces ONLY the `/api/` portion of the documentation site; the
# narrative documentation is built separately with MyST (see docs/myst/).
# The two outputs are combined and deployed by .github/workflows/docs.yml.
#
# Local build:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl
# Output lands in docs/build/ (open docs/build/index.html).
# ---------------------------------------------------------------------------

using Documenter
using GeneralizedGradients

makedocs(
  modules  = [GeneralizedGradients],
  sitename = "GeneralizedGradients.jl",
  authors  = "David Sagan and contributors",
  repo     = Documenter.Remotes.GitHub("bmad-sim", "GeneralizedGradients.jl"),
  format = Documenter.HTML(
    # Pretty (directory) URLs only on CI; flat .html files locally so the build
    # is browsable without a web server.
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical  = "https://bmad-sim.github.io/GeneralizedGradients.jl/api",
    edit_link  = "main",
    assets     = String[],
  ),
  pages = [
    "API Reference" => "index.md",
  ],
  # Ensure every exported symbol has its docstring shown somewhere on the site.
  checkdocs = :exports,
  # Be lenient on the first setup: report problems (broken @refs, missing docs)
  # as warnings rather than failing the build.
  warnonly = true,
)

# NOTE: deployment is handled by the docs workflow (peaceiris/actions-gh-pages),
# which merges this Documenter output with the MyST site, so there is no
# `deploydocs(...)` call here.
