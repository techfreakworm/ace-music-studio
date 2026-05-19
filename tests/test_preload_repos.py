"""Sanity check on the preload list app.py advertises to HF Spaces.

If this drifts the README's ``preload_from_hub`` frontmatter and the
symlink loop in ``_symlink_snapshots_into_models()`` will fall out of
sync — first-user latency on Spaces regresses without anyone noticing
until prod. Cheap to assert here.
"""

from __future__ import annotations


def test_preload_repos_shape():
    from app import _PRELOAD_REPOS

    assert isinstance(_PRELOAD_REPOS, tuple)
    assert len(_PRELOAD_REPOS) == 5
    for repo_id in _PRELOAD_REPOS:
        assert isinstance(repo_id, str)
        assert repo_id.startswith(("ACE-Step/", "Qwen/")), repo_id
