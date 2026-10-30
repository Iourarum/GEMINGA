"""Sphinx configuration for the GEMINGA documentation."""
import datetime, os, sys

sys.path.insert(0, os.path.abspath("../python"))

project = "GEMINGA"
author = "The GEMINGA authors"
copyright = f"{datetime.date.today().year}, {author}"
release = version = "0.18.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["geminga._core", "h5py", "zarr", "tifffile", "scipy", "pyarrow",
                        "matplotlib", "anndata"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "GEMINGA"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo-128.png"
html_css_files = ["geminga.css"]
html_show_sourcelink = True

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2E77C4",
        "color-brand-content": "#2E77C4",
        "color-admonition-background": "#FFFCF3",
        "font-stack": "ui-sans-serif, system-ui, 'Segoe UI', Inter, Roboto, Helvetica, Arial, sans-serif",
        "font-stack--monospace": "'DejaVu Sans Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4CC9F0",
        "color-brand-content": "#4CC9F0",
        "color-background-primary": "#0B2545",
        "color-background-secondary": "#10294C",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/<user>/geminga",
            "html": '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>',
            "class": "",
        },
    ],
}
