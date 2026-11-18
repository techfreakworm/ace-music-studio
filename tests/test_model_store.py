"""Tests for the pinned model manifest and the model store resolver.

Everything here is offline: no weights, no network, and every cache root and
checkpoints directory lives under ``tmp_path``. The one test that downloads
(two small files) is marked ``network`` and is deselected by default.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "model_manifest.json"

UMBRELLA = "ACE-Step/Ace-Step1.5"
XL_SFT = "ACE-Step/acestep-v15-xl-sft"
QWEN = "Qwen/Qwen2.5-7B-Instruct"


def _raw_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _by_repo(manifest: dict) -> dict[str, dict]:
    return {r["repo_id"]: r for r in manifest["repos"]}


# --- R1: manifest ------------------------------------------------------------


def test_manifest_schema_and_fields():
    m = _raw_manifest()
    assert m["schema"] == 1
    assert isinstance(m["repos"], list) and m["repos"]
    keys = [r["key"] for r in m["repos"]]
    assert len(keys) == len(set(keys)), "duplicate repo keys"
    repo_ids = [r["repo_id"] for r in m["repos"]]
    assert len(repo_ids) == len(set(repo_ids)), "duplicate repo ids"
    for r in m["repos"]:
        assert re.fullmatch(r"[0-9a-f]{40}", r["revision"]), r
        assert r["startup"] in {"required", "optional", "lazy"}, r
        assert r["files"], r
        paths = [f["path"] for f in r["files"]]
        assert len(paths) == len(set(paths)), f"duplicate paths in {r['repo_id']}"
        for f in r["files"]:
            assert isinstance(f["size"], int) and f["size"] > 0, f
            assert not f["path"].startswith("/") and ".." not in f["path"].split("/"), f
            assert not any(ch in f["path"] for ch in "*?[]"), f"glob character in {f['path']}"


def test_manifest_pins_the_expected_repos_and_commits():
    repos = _by_repo(_raw_manifest())
    expected = {
        UMBRELLA: ("19671f406d603126926c1b7e2adc169acbcade22", "required"),
        XL_SFT: ("d06de46b4622f781cf07f4a013a67d591ca52819", "required"),
        QWEN: ("a09a35458c702b33eeacc393d103063234e8bc28", "optional"),
        "ACE-Step/ACE-Step-v1-chinese-rap-LoRA": ("e40a223cb69a60ba8a8c46752987c152b8a5f828", "lazy"),
        "ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA": ("cb829a12775740c830a6d49795f16913065dc492", "lazy"),
    }
    assert set(repos) == set(expected)
    for repo_id, (revision, startup) in expected.items():
        assert repos[repo_id]["revision"] == revision, repo_id
        assert repos[repo_id]["startup"] == startup, repo_id


def test_manifest_umbrella_has_components_but_no_turbo_weights():
    umbrella = _by_repo(_raw_manifest())[UMBRELLA]
    paths = {f["path"] for f in umbrella["files"]}
    turbo = {p for p in paths if p.startswith("acestep-v15-turbo/")}
    assert turbo == {"acestep-v15-turbo/config.json"}
    for component in ("vae/", "Qwen3-Embedding-0.6B/", "acestep-5Hz-lm-1.7B/"):
        assert any(p.startswith(component) for p in paths), component
    assert "vae/diffusion_pytorch_model.safetensors" in paths
    assert "Qwen3-Embedding-0.6B/model.safetensors" in paths
    assert "acestep-5Hz-lm-1.7B/model.safetensors" in paths
    # Top-level files of the umbrella are never read by the fork.
    assert not any("/" not in p for p in paths)


def test_manifest_xl_sft_and_qwen_files():
    repos = _by_repo(_raw_manifest())
    xl = {f["path"] for f in repos[XL_SFT]["files"]}
    assert xl == {
        "apg_guidance.py",
        "config.json",
        "configuration_acestep_v15.py",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "modeling_acestep_v15_xl_base.py",
        "silence_latent.pt",
    }
    qwen = {f["path"] for f in repos[QWEN]["files"]}
    assert qwen == {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
