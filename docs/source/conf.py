
import pathlib
import sys
TOP = pathlib.Path(__file__).parent.parent.absolute()
if str(TOP) not in sys.path:
    sys.path.insert(0, str(TOP))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ndtorch'
copyright = '2023, Ivan Morozov'
author = 'Ivan Morozov'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosectionlabel',
              'nbsphinx'
              ]
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc Configuration ---------------------------------------------------

# Add here all modules to be mocked up. When the dependencies are not met
# at building time. Here used to have PyQT mocked.
autodoc_mock_imports = ['PyQt5', 'PyQt5.QtGui', 'PyQt5.QtCore', 'PyQt5.QtWidgets', "matplotlib.backends.backend_qt5agg"]
