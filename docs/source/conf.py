import os
import sys

# Permet à Sphinx de trouver ton package
sys.path.insert(0, os.path.abspath('../..'))

project   = 'NEUREQUA'
author    = 'Dornier V.'
release   = '0.1'

extensions = [
    'autoapi.extension',      # extraction auto des docstrings
    'sphinx.ext.napoleon',    # support NumPy/Google docstring style
    'sphinx.ext.viewcode',    # lien "voir le code source"
    'sphinx.ext.mathjax',     # rendu des équations LaTeX
]

# AutoAPI : pointe vers ton dossier de code
autoapi_dirs = ['../../NEUREQUA']   # adapte selon ton arborescence
autoapi_type = 'python'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]

# Thème
html_theme = 'furo'
html_static_path = ['_static']
html_logo = "logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# Support des équations
mathjax_path = (
    'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
)
