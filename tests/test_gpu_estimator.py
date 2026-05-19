"""Unit tests for the per-mode GPU duration extraction."""

from __future__ import annotations


def test_extract_generate_positional():
    from app import _extract_duration_s

    args = ("a prompt", "lyrics body", 45.0, "vocal_male", None)
    kwargs = {}
    assert _extract_duration_s("generate", args, kwargs) == 45.0


def test_extract_cover_at_index_3():
    from app import _extract_duration_s

    args = ("ref.wav", "new style", "new lyrics", 60.0)
    kwargs = {}
    assert _extract_duration_s("cover", args, kwargs) == 60.0


def test_extract_extend_uses_extra_duration_s_kwarg():
    from app import _extract_duration_s

    args = ("seed.wav", "more of the same", "extension lyrics", 25.0)
    kwargs = {}
    assert _extract_duration_s("extend", args, kwargs) == 25.0


def test_extract_extend_kwarg_form():
    from app import _extract_duration_s

    assert _extract_duration_s("extend", (), {"extra_duration_s": 18.5}) == 18.5


def test_extract_edit_segment_window():
    from app import _extract_duration_s

    args = ("src.wav", "repaint", "src lyrics", "new lyrics", 10.0, 22.5)
    kwargs = {}
    assert _extract_duration_s("edit", args, kwargs) == 12.5


def test_extract_edit_kwarg_window():
    from app import _extract_duration_s

    kwargs = {"segment_start_s": 5.0, "segment_end_s": 20.0}
    assert _extract_duration_s("edit", (), kwargs) == 15.0


def test_extract_lyrics_returns_none():
    from app import _extract_duration_s

    assert _extract_duration_s("lyrics", ("brief", "ABAB"), {}) is None


def test_extract_generate_falls_back_when_missing():
    from app import _extract_duration_s

    # No positional duration, no kwarg → None
    assert _extract_duration_s("generate", ("p", "l"), {}) is None


def test_estimator_clamp_floor():
    from app import _estimate_gpu_duration

    # lyrics base=15 + 1.0*2 = 17 → clamped up to 60s floor.
    assert _estimate_gpu_duration("lyrics", {"duration_s": 1.0}) == 60


def test_estimator_clamp_ceiling():
    from app import _estimate_gpu_duration

    # 240s requested * 2 = 480 + base 30 = 510 → clamped to 300
    assert _estimate_gpu_duration("generate", {"duration_s": 240}) == 300


def test_estimator_mode_specific_base():
    from app import _estimate_gpu_duration

    # 30s requested * 2 = 60 + base 40 (cover) = 100s
    assert _estimate_gpu_duration("cover", {"duration_s": 30}) == 100
