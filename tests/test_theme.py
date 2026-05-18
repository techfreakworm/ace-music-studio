"""L1 theme assertions — palette tokens, CSS presence."""

from __future__ import annotations

import theme


def test_palette_tokens_are_brutalist_mono():
    assert theme.BG == "#0A0A0A"
    assert theme.INK == "#E5E5E5"
    assert theme.PRIMARY == "#FFFFFF"
    # No color accent — that's the whole point of Brutalist Mono
    for v in (
        theme.BG,
        theme.SURFACE,
        theme.SURFACE_STRONG,
        theme.BORDER,
        theme.BORDER_STRONG,
        theme.INK,
        theme.INK_MUTED,
        theme.PRIMARY,
    ):
        assert v.startswith("#")
        assert len(v) == 7  # all hex, no rgba


def test_css_contains_responsive_breakpoints():
    css = theme.CSS
    assert "@media" in css
    assert "1024px" in css  # tablet breakpoint
    assert "640px" in css  # mobile breakpoint


def test_build_theme_returns_gradio_theme():
    import gradio as gr

    t = theme.build_theme()
    # gr.themes.Base is the parent class
    assert isinstance(t, gr.themes.Base)
