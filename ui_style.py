"""One button palette for the whole app.

Every tool used to carry its own copy of these rules, which is how a new tool
ended up looking unlike the File Format Converter beside it. The palette lives
in one place now so the tools cannot drift apart again.
"""

import streamlit as st

PRIMARY, PRIMARY_HOVER = "#3b82f6", "#2563eb"
DOWNLOAD, DOWNLOAD_HOVER = "#0e4d92", "#0a3a6e"
QUIET, QUIET_HOVER = "#9ca3af", "#868e96"


def _rule(selector, colour):
    return (f"{selector} {{ background-color: {colour}; border-color: {colour}; "
            "color: #ffffff; }")


def apply_button_style(quiet_keys=()):
    """Blue for the action buttons, a deeper blue for the file downloads.

    ``quiet_keys`` names the primary buttons that should stay light grey
    instead — the "remove what I uploaded" kind, which has no business
    competing with the action the user actually came for. The rule is keyed on
    the button's own ``key``, so it is more specific than the blue one above
    and wins.
    """
    primary = 'button[data-testid^="stBaseButton-primary"]'
    download = '[data-testid="stDownloadButton"] button'
    rules = [
        _rule(primary, PRIMARY),
        _rule(f"{primary}:hover", PRIMARY_HOVER),
        _rule(download, DOWNLOAD),
        _rule(f"{download}:hover", DOWNLOAD_HOVER),
    ]
    for key in quiet_keys:
        rules.append(_rule(f".st-key-{key} {primary}", QUIET))
        rules.append(_rule(f".st-key-{key} {primary}:hover", QUIET_HOVER))
    st.markdown("<style>\n" + "\n".join(rules) + "\n</style>",
                unsafe_allow_html=True)
