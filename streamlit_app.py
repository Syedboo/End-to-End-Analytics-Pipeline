"""Streamlit Community Cloud entrypoint.

The production dashboard lives in appstreamlit.py. Keeping this lightweight
wrapper lets Streamlit Community Cloud use its conventional default filename
without duplicating dashboard code.
"""

import appstreamlit  # noqa: F401  # Imported for Streamlit side effects.
