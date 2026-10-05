"""The static release-asset validator passes on the repository and refuses representative defects."""

# ruff: noqa: E501

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_release_assets", ROOT / "tools" / "validate_release_assets.py")
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_repository_passes_static_validation():
    assert "notebook+carrier+lock+parity" in validator.validate_all()


def test_model_card_declares_spec_1_2_and_all_required_sections():
    validator.validate_model_card()
    text = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert 'model_card_spec: "1.2"' in text
    assert "<!--" not in text, "no tooltip or placeholder comment may survive (G9, G11)"


def test_status_stays_candidate_without_hosted_evidence():
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status
    assert "| Candidate |" in (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")


def test_placeholder_detection(tmp_path):
    with pytest.raises(validator.ValidationError):
        validator._check(not validator.PLACEHOLDER.search("TODO: write"), "placeholder")
