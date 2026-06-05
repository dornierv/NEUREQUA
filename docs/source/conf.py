# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NEUREQUA'
copyright = '2026, Dornier V.'
author = 'Dornier V.'
release = '0.1..1.19'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',      # extraction auto des docstrings
    'sphinx.ext.napoleon',    # support NumPy/Google docstring style
    'sphinx.ext.viewcode',    # lien "voir le code source"
    'sphinx.ext.mathjax',     # rendu des équations LaTeX
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
