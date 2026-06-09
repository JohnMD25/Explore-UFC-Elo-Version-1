"""Pytest configuration.

Mocks Streamlit UI add-on packages that app.py imports at module level
(streamlit_shadcn_ui, streamlit_echarts, etc.). These add-ons are not
needed for pure-function unit tests, so we substitute MagicMocks. This
lets the test suite run in any Python environment that has streamlit +
pandas + numpy installed, without requiring every UI add-on.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Any module imported at the top level of app.py that isn't pure-Python data
# tooling. Add new entries here if app.py grows new add-on imports.
_MOCKED_MODULES = [
    "streamlit_shadcn_ui",
    "streamlit_echarts",
    "streamlit_option_menu",
    "streamlit_extras",
    "streamlit_extras.stylable_container",
    "streamlit_image_zoom",
    "st_aggrid",
    "streamlit_aggrid",
]

for _name in _MOCKED_MODULES:
    sys.modules.setdefault(_name, MagicMock())