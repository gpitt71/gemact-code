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

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = 'GEMAct'
copyright = '2022, Gabriele Pittarello, Edoardo Luini, Manfred Marvin Marchione'
author = 'Gabriele Pittarello, Edoardo Luini, Manfred Marvin Marchione'

# The full version, including alpha/beta/rc tags
release = '2022'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_sitemap',
    'nbsphinx',
    'sphinxcontrib.bibtex'
]

# bibliography
exclude_patterns = ['_build']
bibtex_bibfiles = ['refs.bib']
sitemap_filename = "sitemap.xml"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_baseurl = 'https://github.com/GEM-analytics/gemact/'
html_theme = 'sphinx_book_theme'
html_title = "gemact"
html_logo = "images/GEMActlogo.png"
html_favicon = "images/webtab.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# lasciare unicamente il logo
html_theme_options = {
  "logo_only": True,
   "home_page_in_toc": True,
   "repository_url": "https://github.com/gpitt71/gemact-code",
   "use_repository_button": True,
    "use_edit_page_button": True
}

add_module_names = False