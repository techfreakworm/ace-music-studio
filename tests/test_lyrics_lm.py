"""L2 tests for lyrics LM — generation is mocked at the model boundary.

The real Qwen 2.5 7B model is never loaded in CI. We only verify the prompt
shape, the call boundary to ``_get_lm()``, and the normalisation pass that
lowercases section tags before returning to the caller.
"""

from __future__ import annotations

import socket
import sys
import types
from unittest.mock import MagicMock

import pytest

import lyrics_lm as ll
import model_store


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


# --- where transformers loads Qwen from (no Hub calls in the GPU worker on Spaces) ---

QWEN_REV = "c" * 40


class _Recorder:
    """Fake ``from_pretrained`` target that records its arguments."""

    calls: list

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        obj = MagicMock()
        obj.to.return_value = obj
        return obj


@pytest.fixture
def fake_hf(monkeypatch):
    """Stand-in torch and transformers so ``_load_hflm`` runs without real weights."""

    Tok = type("Tok", (_Recorder,), {"calls": []})
    Model = type("Model", (_Recorder,), {"calls": []})
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"
    fake_torch.float32 = "fp32"
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = Tok
    fake_transformers.AutoModelForCausalLM = Model
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return Tok, Model


@pytest.fixture
def qwen_store(monkeypatch, tmp_path):
    """A one-repo manifest and tmp cache roots for the model store."""
    files = {"config.json": 5, "model-00001-of-00001.safetensors": 9, "tokenizer.json": 3}
    entry = {
        "key": "qwen",
        "repo_id": ll._DEFAULT_CUDA_ID,
        "revision": QWEN_REV,
        "startup": "optional",
        "files": [{"path": p, "size": s} for p, s in files.items()],
    }
    roots = [tmp_path / "hub", tmp_path / "app-cache", tmp_path / "tmp-cache"]
    monkeypatch.setattr(model_store, "load_manifest", lambda *a, **k: {"schema": 1, "repos": [entry]})
    monkeypatch.setattr(model_store, "lookup_roots", lambda: roots)

    def _write(root):
        snap = model_store.snapshot_path(root, entry)
        for path, size in files.items():
            (snap / path).parent.mkdir(parents=True, exist_ok=True)
            (snap / path).write_bytes(b"q" * size)
        return snap

    return roots, _write


@pytest.fixture
def no_network(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


def test_load_hflm_on_spaces_uses_the_local_snapshot(monkeypatch, fake_hf, qwen_store, no_network):
    roots, write = qwen_store
    snap = write(roots[1])
    monkeypatch.setenv("SPACE_ID", "test/test")
    tok, model = fake_hf

    ll._load_hflm("cuda")

    assert tok.calls == [(str(snap), {})]
    assert model.calls == [(str(snap), {"torch_dtype": "bf16"})]


def test_load_hflm_on_spaces_without_a_local_snapshot_raises_value_error(
    monkeypatch, fake_hf, qwen_store, no_network
):
    roots, write = qwen_store
    snap = write(roots[0])
    (snap / "tokenizer.json").write_bytes(b"short")  # wrong size: not complete
    monkeypatch.setenv("SPACE_ID", "test/test")
    monkeypatch.setattr(ll, "_LM", None)
    monkeypatch.setattr(ll.ap, "detect_device", lambda: "cuda")
    tok, model = fake_hf

    with pytest.raises(ValueError, match=r"Qwen2\.5-7B-Instruct") as exc:
        ll._get_lm()

    # No 15 GB download inside the GPU call: nothing is loaded by repo id.
    assert tok.calls == [] and model.calls == []
    assert "restart" in str(exc.value).lower()
    assert ll._LM is None  # the next click looks again


def test_load_hflm_off_spaces_is_unchanged(monkeypatch, fake_hf, qwen_store):
    roots, write = qwen_store
    write(roots[0])  # even with a local snapshot, off Spaces nothing changes
    monkeypatch.delenv("SPACE_ID", raising=False)
    tok, model = fake_hf

    ll._load_hflm("cpu")

    assert tok.calls == [(ll._DEFAULT_CUDA_ID, {})]
    assert model.calls == [(ll._DEFAULT_CUDA_ID, {"torch_dtype": "fp32"})]


def test_mlx_path_is_unchanged(monkeypatch):
    loaded = []
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda repo: loaded.append(repo) or ("model", "tokenizer")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr(ll.ap, "detect_device", lambda: "mps")
    monkeypatch.setattr(model_store, "local_snapshot", MagicMock(side_effect=AssertionError("store used")))

    lm = ll._load_lm()

    assert loaded == [ll._DEFAULT_MAC_ID]
    assert isinstance(lm, ll._MLXLM)
