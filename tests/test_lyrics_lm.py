"""L2 tests for lyrics LM — generation is mocked at the model boundary.

The real Qwen 2.5 7B model is never loaded in CI. We only verify the prompt
shape, the call boundary to ``_get_lm()``, and the normalisation pass that
lowercases section tags before returning to the caller.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import lyrics_lm as ll


def test_build_system_prompt_includes_tag_format():
    sp = ll.build_system_prompt()
    low = sp.lower()
    assert "[verse" in low
    assert "[chorus" in low


def test_generate_lyrics_calls_lm_and_returns_text(monkeypatch):
    fake_lm = MagicMock()
    fake_lm.generate.return_value = "[verse] x\n[chorus] y\n"
    monkeypatch.setattr(ll, "_get_lm", lambda: fake_lm)

    out = ll.generate_lyrics(
        brief="a song",
        structure="intro, verse, chorus, outro",
        language="en",
        tone="upbeat",
        verse_lines=4,
        chorus_lines=4,
        bridge_lines=2,
        rhyme="loose",
        temperature=0.85,
        top_p=0.9,
        top_k=40,
        max_new_tokens=200,
        seed=42,
    )
    assert "[verse]" in out
    fake_lm.generate.assert_called_once()


def test_normalise_lyrics_lowercases_tags():
    norm = ll._normalise(" [Verse 1]\nhello\n[Chorus]\nworld ")
    assert "[verse 1]" in norm
    assert "[chorus]" in norm
    assert "[Verse" not in norm
