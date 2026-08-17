"""One console line whenever a user downloads a file.

The app itself says nothing about what visitors actually take away, so every
download button reports the tool it belongs to and when it happened. Used as
the ``on_click`` callback of ``st.download_button``, which Streamlit runs when
the button is pressed.
"""

from datetime import datetime


def log_download(tool, what="Converted files"):
    """Print e.g. ``Converted files downloaded — Plotting — 2026-08-17 14:35``."""
    print(f"{what} downloaded — {tool} — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
