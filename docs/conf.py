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
sys.path.insert(0, os.path.abspath('./../'))
# Path to venv on windows machine
sys.path.insert(0, os.path.abspath('./../virtualenv/Lib/site-packages'))

import version as version_config

# -- Project information -----------------------------------------------------

project = 'Uncertainty Wizard'
copyright = 'Michael Weiss and Paolo Tonella at the Università della Svizzera Italiana. License: MIT'
author = 'Michael Weiss'

# The full version, including alpha/beta/rc tags
# version = '0.14'
version = version_config.VERSION
release = version_config.RELEASE


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Used to generate documentation from docstrings
    'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    # Check for Documentation Coverage
    'sphinx.ext.coverage',
    # Enable understanding of various documentation styles
    'sphinx.ext.napoleon',
    # Allow to use markdown files
    'recommonmark',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Disabled for Double Blindness
# TODO check main/master branch name
html_context = {
  'display_github': True,
  'github_user': 'testingautomated-usi',
  'github_repo': 'uncertainty_wizard',
  'github_version': 'master/docs/',
}
