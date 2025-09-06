# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from unittest.mock import MagicMock

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../gaia")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Mock problematic modules before any imports
class MockTorch(MagicMock):
    """Mock torch module"""
    device = MagicMock()
    nn = MagicMock()
    optim = MagicMock()
    
sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Mock the DEVICE constant specifically
from _mock_device import DEVICE, device
sys.modules['gaia.core.device'] = MagicMock()
sys.modules['gaia.utils.device'] = MagicMock()


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GAIA Framework'
copyright = '2025, Juan Zambrano, Sridhar Mahadevan, Enrique ter Horst'
author = 'Juan Zambrano, Sridhar Mahadevan, Enrique ter Horst'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
    'inherited-members': True
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_mock_imports = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'numpy',
    'scipy',
    'sklearn',
    'matplotlib',
    'seaborn',
    'pandas',
    'networkx',
    'sympy',
    'tqdm',
    'wandb',
    'tensorboard',
    'transformers',
    'datasets',
    'accelerate',
    'gaia.core.DEVICE',
    'gaia.core.device',
    'gaia.utils.device'
]
autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Math settings
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'macros': {
            'RR': '{\\mathbb{R}}',
            'CC': '{\\mathbb{C}}',
            'NN': '{\\mathbb{N}}',
            'ZZ': '{\\mathbb{Z}}',
            'QQ': '{\\mathbb{Q}}',
            'Set': '{\\mathbf{Set}}',
            'Cat': '{\\mathbf{Cat}}',
            'Hom': '{\\text{Hom}}',
            'id': '{\\text{id}}',
            'op': '{\\text{op}}',
        }
    }
}

# Todo settings
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
try:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Theme options for premium branding
html_theme_options = {
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#1F3B82',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'custom.css',
]

# HTML context for enhanced branding
html_context = {
    'display_github': False,
    'display_bitbucket': False,
    'display_gitlab': False,
}

# Enhanced HTML options
html_title = 'GAIA Framework Documentation'
html_short_title = 'GAIA Framework'
html_logo = '_static/gaia_logo.svg'
html_favicon = '_static/gaia_favicon.ico'

# Enhanced sidebar
html_sidebars = {
    '**': [
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}

# Additional theme customizations
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

# Custom footer
html_last_updated_fmt = '%b %d, %Y'
html_use_smartypants = True
