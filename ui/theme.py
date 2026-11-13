"""The approved editorial visual language, applied to native controls."""

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
PINE = "#25664f"
ORANGE = "#a74f2c"


def apply_theme() -> None:
    fonts = (ASSETS / "fonts.css").read_text(encoding="utf-8")
    styles = (ASSETS / "theme.css").read_text(encoding="utf-8")
    st.html(f"<style>{fonts}\n{styles}</style>")
