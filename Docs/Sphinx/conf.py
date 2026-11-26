# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Schola'
copyright = "Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All Rights Reserved."
author = 'Advanced Micro Devices'

import sys
import os
sys.path.insert(0, os.path.abspath("../../Resources/python"))
sys.path.append(os.path.abspath("./_ext"))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_tabs.tabs',
    'breathe',
    "blueprint",
    "localref",
]

# autosectionlabel_prefix_document = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

#code highlighting
pygments_style = 'sphinx'

# Generate the API documentation when building
autosummary_generate=True

# -- Options for autodoc extension -------------------------------------------
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_book_theme"
#html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = './_static/AMD_Schola_Lockup_RGB_Blk.png'
html_css_files = ['schola_theme.css']
html_favicon = "./_static/icon.png"

# disable index page
html_domain_indices = False

html_theme_options = {
    # sphinx_book_theme options
    'show_navbar_depth': 1,
    'show_toc_level': 2,
    'navigation_with_keys': False,
    'home_page_in_toc': True,
}

PYTHON_VERSION = "Python 3.11 to 3.12 "
UNREAL_VERSION = "Unreal Engine 5.7"
UNREAL_VERSION_EXACT = "5.7.3"

rst_prolog =f"""
.. |py_version| replace:: {PYTHON_VERSION} 
.. _`py_version`: https://www.python.org/downloads/

.. |unreal_version| replace:: ({UNREAL_VERSION_EXACT} is recommended for reproducibility)

.. |unreal_version_win| replace:: {UNREAL_VERSION}
.. _`unreal_version_win`: https://www.unrealengine.com/en-US/download 

.. |unreal_version_linux| replace:: {UNREAL_VERSION}
.. _`unreal_version_linux`: https://www.unrealengine.com/en-US/download


.. |ubuntu_version| replace:: Ubuntu 22.04
.. _`ubuntu_version`: https://releases.ubuntu.com/jammy/
.. |ubuntu_version_exact| replace:: (22.04.4 Desktop x86 64-bit is recommended for reproducibility)

.. |windows_11| replace:: Windows 11
.. _`windows_11`: https://www.microsoft.com/en-us/software-download/windows11

.. |windows_10| replace:: Windows 10
.. _`windows_10`: https://www.microsoft.com/en-us/software-download/windows10

.. |vs_version| replace:: (Visual Studio Professional 2022 (64-bit) - LTSC 17.8 is recommended for reproducibility)
"""

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True

# -- Options for breathe extension -------------------------------------------
# https://breathe.readthedocs.io/en/latest/directives.html#config-values

# most of these are set to the defaults but exposed here to make it easier to track
#breathe_build_directory= ...
breathe_implementation_filename_extensions = ['.c', '.cc', '.cpp']
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True
breathe_show_include = True
breathe_projects = {
    "Schola" : "../Doxygen/xml"
}
breathe_default_project="Schola"
#Make the doxygen classes actually display things
breathe_default_members = ('members', 'undoc-members')

# -- Options for sphinx_tabs extension ---------------------------------------
sphinx_tabs_disable_tab_closing = True

# -- Options for intersphinx extension ---------------------------------------
USE_INTERSPHINX = True # quick toggle to speed up build time while debugging
if USE_INTERSPHINX:
    intersphinx_mapping = {
        'python': ('https://docs.python.org/3', None), 
        'ray': ('https://docs.ray.io/en/master/', None),
        'gym': ('https://gymnasium.farama.org/', None),
        'stable_baselines3': ('https://stable-baselines3.readthedocs.io/en/master/', None),
        "cyclopts": ('https://cyclopts.readthedocs.io/en/v4.2.2/', None)
        }

# -- Options for blueprints extension -----------------------------------------
blueprint_dir = './blueprints'
