"""Small presentation helpers shared by Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from .config import STYLES_PATH


def inject_styles(path: str | Path = STYLES_PATH) -> None:
    """Load the app stylesheet when it is available."""
    css_path = Path(path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def setup_page(page_title: str) -> None:
    """Apply the shared stylesheet to a page managed by app.py navigation."""
    # Page configuration is owned by app.py through st.navigation.
    inject_styles()


def page_header(kicker: str, title: str, description: str) -> None:
    """Render the shared page title block."""
    import html

    st.markdown(
        '<div class="hero">'
        f'<div class="hero-kicker">{html.escape(kicker)}</div>'
        f'<h1>{html.escape(title)}</h1>'
        f'<div class="hero-copy">{html.escape(description)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def format_percent(value: float, decimals: int = 1) -> str:
    return "—" if value is None else f"{value:.{decimals}f}%"


def format_currency(value: float, currency: str = "$") -> str:
    return "—" if value is None else f"{currency}{value:,.0f}"


def humanize_feature(name: str) -> str:
    """Convert an encoded feature name to a readable label."""
    return name.replace("categorical__", "").replace("remainder__", "").replace("_", " ").strip().title()
