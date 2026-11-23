# Configuration file for the Sphinx documentation builder.
#
# Builds the halogal docs on Read the Docs (see ../.readthedocs.yaml) and in
# the GitHub "Docs" workflow (../.github/workflows/docs.yml).
#
#   * Tutorials: the Jupyter notebooks in ../examples and ../notebooks are
#     copied into docs/tutorials/ at build time and rendered by myst-nb from
#     their *stored* outputs (they are not re-executed), so re-run and save a
#     notebook to refresh its page.
#   * API reference: generated from the docstrings by sphinx.ext.autosummary
#     (recursive over the whole ``halogal`` package) + napoleon for the
#     NumPy-style sections.
#
# Local build:  pip install -r docs/requirements.txt -e . && make -C docs html

import os
import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

import halogal  # noqa: E402

# -- Project information -----------------------------------------------------

project = "halogal"
author = "Marko Shuntov"
copyright = "2025, Marko Shuntov"
release = halogal.__version__
version = ".".join(release.split(".")[:2])

# -- Tutorial notebooks ------------------------------------------------------
# Sphinx can only see files under docs/, so mirror the notebooks into
# docs/tutorials/ (git-ignored) on every build. Order here = order in the
# sidebar; the page title comes from each notebook's first heading.

TUTORIALS = [
    "examples/basic_usage.ipynb",
    "examples/mstar_hod_tutorial.ipynb",
    "examples/ffb_hod_tutorial.ipynb",
    "examples/inference_tutorial.ipynb",
    "notebooks/validate_shuntov25.ipynb",
]

_tut_dir = DOCS / "tutorials"
if _tut_dir.exists():
    shutil.rmtree(_tut_dir)
_tut_dir.mkdir()
for _nb in TUTORIALS:
    shutil.copy2(ROOT / _nb, _tut_dir / Path(_nb).name)

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_nb",  # Markdown pages (inference.md, ...) and .ipynb notebooks
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
templates_path = ["_templates"]

# -- autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
# Optional sampler / plotting backends are imported lazily inside methods;
# mock them so the docs never depend on them being installed.
autodoc_mock_imports = ["emcee", "dynesty", "corner", "h5py", "schwimmbad"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
# Render "Attributes" sections as fields, so attributes that are also
# properties (e.g. HMRModel_FFB.M_ffb) aren't documented twice.
napoleon_use_ivar = True

# -- myst-nb -----------------------------------------------------------------

nb_execution_mode = "off"  # render the outputs saved in the notebooks
nb_merge_streams = True
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
# The notebooks jump from their H1 title straight to H3 in places.
suppress_warnings = ["myst.header"]

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "halomod": ("https://halomod.readthedocs.io/en/latest/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_context = {
    "display_github": True,
    "github_user": os.environ.get("DOCS_GITHUB_USER", "mShuntov"),
    "github_repo": "uvlf-hod",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
html_static_path = []
