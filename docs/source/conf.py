# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme, sphinx_autodoc_typehints
from typing import List

sys.path.insert(0, os.path.abspath("../../mdgo"))

# -- Project information -----------------------------------------------------

project = "mdgo"
copyright = "2021, Tingzheng Hou"
author = "Tingzheng Hou"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
]

source_suffix = [".rst"]
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/mdgo-white.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

autodoc_typehints = "description"

autodoc_mock_imports = [
    "typing_extensions",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "tqdm",
    "monty",
    "pymatgen",
    "statsmodels",
    "pubchempy",
    "MDAnalysis",
    "selenium",
    "matplotlib.pyplot",
    "MDAnalysis.lib",
    "MDAnalysis.lib.distances",
    "MDAnalysis.analysis",
    "MDAnalysis.analysis.msd",
    "MDAnalysis.analysis.distances",
    "pymatgen.io",
    "pymatgen.io.lammps",
    "pymatgen.io.lammps.data",
    "statsmodels.tsa",
    "statsmodels.tsa.stattools",
    "scipy.signal",
    "scipy.optimize",
    "selenium.common",
    "selenium.common.exceptions",
    "selenium.webdriver",
    "selenium.webdriver.support",
    "selenium.webdriver.support.ui",
    "selenium.webdriver.common",
    "selenium.webdriver.common.by",
]
