# Configuration file for the Sphinx documentation builder.
#
# Narrative documentation, written in MyST Markdown and rendered by Sphinx with
# the Furo theme. The API reference is built separately by Documenter
# (docs/api/make.jl) and served at /api/; docs/build.py combines the two.

import os

# -- Project information -----------------------------------------------------
project = 'GeneralizedGradients.jl'
copyright = '2026, GeneralizedGradients.jl Contributors'
author = 'David Sagan and contributors'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',              # MyST Markdown
    'sphinx.ext.githubpages',   # emit .nojekyll for GitHub Pages
    'sphinx.ext.intersphinx',   # cross-reference into the Documenter API
    'sphinx.ext.mathjax',
]

numfig = True

# -- Intersphinx: let prose link into the Documenter API ---------------------
# Documenter writes an objects.inv into docs/api/build (built first by
# docs/build.py). The fixer below rewrites the absolute API URLs to site-
# relative paths so previews and local builds work.
_docs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_api_base_url = 'https://bmad-sim.github.io/GeneralizedGradients.jl/api/'
intersphinx_mapping = {
    'julia': (_api_base_url,
              (os.path.join(_docs_dir, 'api', 'build', 'objects.inv'),)),
}
_api_subpath = 'api/'


def _fix_intersphinx_refs(app, doctree, docname):
    """Rewrite intersphinx absolute API URLs to relative paths for local/preview browsing."""
    from docutils import nodes
    from posixpath import relpath, dirname

    for node in doctree.traverse(nodes.reference):
        uri = node.get('refuri', '')
        if not uri.startswith(_api_base_url):
            continue
        rel_part = uri[len(_api_base_url):]
        target = _api_subpath + rel_part
        doc_dir = dirname(docname)
        if '#' in target:
            path_part, fragment = target.split('#', 1)
            node['refuri'] = relpath(path_part, doc_dir) + '#' + fragment
        else:
            node['refuri'] = relpath(target, doc_dir)


# -- Minimal Julia domain so intersphinx can resolve Documenter's jl:* roles --
from sphinx.domains import Domain, ObjType
from sphinx.roles import XRefRole


class _JuliaDomain(Domain):
    name = 'jl'
    label = 'Julia'
    object_types = {
        'function': ObjType('function', 'function'),
        'method':   ObjType('method', 'method'),
        'type':     ObjType('type', 'type'),
        'macro':    ObjType('macro', 'macro'),
        'module':   ObjType('module', 'module'),
    }
    roles = {
        'function': XRefRole(),
        'method':   XRefRole(),
        'type':     XRefRole(),
        'macro':    XRefRole(),
        'module':   XRefRole(),
    }
    directives = {}
    initial_data = {'objects': {}}

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        return None  # intersphinx handles external references

    def get_objects(self):
        return iter([])


def setup(app):
    app.add_domain(_JuliaDomain)
    app.connect('doctree-resolved', _fix_intersphinx_refs)


# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'deflist',
    'colon_fence',
    'linkify',
]
myst_heading_anchors = 3

# -- HTML output (Furo) ------------------------------------------------------
html_theme = 'furo'
html_theme_options = {
    'source_repository': 'https://github.com/bmad-sim/GeneralizedGradients.jl',
    'source_branch': 'main',
    'source_directory': 'docs/src/',
    'navigation_with_keys': True,
    'sidebar_hide_name': False,
}
html_title = 'GeneralizedGradients.jl'
templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = ['custom.css']

# Add the "API Reference →" link to the Furo sidebar on every page.
html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/scroll-start.html',
        'sidebar/navigation.html',
        'sidebar-external-links.html',
        'sidebar/scroll-end.html',
    ]
}
