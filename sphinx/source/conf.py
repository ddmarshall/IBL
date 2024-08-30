# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: skip-file
# type: ignore

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../ibl'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IBL'
copyright = '2023-2024, David D. Marshall'
author = 'David D. Marshall'
release = '0.6.0.dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary'
]

# napoleon options
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# autodoc options
autodoc_inherit_docstrings = True
autodoc_member_order = 'bysource'
autodoc_preserve_defaults = True
autodoc_typehints = 'description'
autodoc_typehits_format = 'short'

# autosummary options
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
python_use_unqualified_type_names = True

# sphinx options
html_copy_source = False
html_show_sourcelink = False

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'


# from readthedocs
html_theme_options = {
    # 'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    # 'analytics_anonymize_ip': False,
    # 'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
