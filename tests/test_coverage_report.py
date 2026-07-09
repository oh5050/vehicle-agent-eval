"""Tests for coverage_report.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.coverage_report import TARGETS, t2_combo_key, t5_pair_stats, t6_stimulus_distribution


def test_t2_combo_key():
    calls = [{"tool": "media_control"}, {"tool": "climate_set"}]
    assert t2_combo_key(calls) == "climate_set + media_control"


def test_t5_pair_stats():
    records = [
        {"utterance_type": "T5", "utterance": "트렁크 열어", "expected": {"kind": "execute"}},
        {"utterance_type": "T5", "utterance": "트렁크 열어", "expected": {"kind": "confirm"}},
        {"utterance_type": "T5", "utterance": "문 잠가", "expected": {"kind": "execute"}},
    ]
    stats = t5_pair_stats(records)
    assert stats["complete_pairs"] == 1
    assert stats["incomplete"] == 1


def test_t6_stimulus_distribution():
    records = [
        {"utterance_type": "T6", "utterance": "좀 춥네", "note": ""},
        {"utterance_type": "T6", "utterance": "졸려", "note": "sleepy drive"},
    ]
    dist = t6_stimulus_distribution(records)
    assert dist["cold"] >= 1
    assert dist["sleepy"] >= 1


def test_targets_sum():
    assert sum(TARGETS.values()) == 120
