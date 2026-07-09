"""Tests for aggregator latency and missing-value handling."""

from __future__ import annotations

from src.aggregator import latency_percentiles, percentile


def test_percentile_empty():
    assert percentile([], 50) is None


def test_latency_percentiles_skips_none_ttft():
    rows = [
        {"utterance_type": "T1", "ttft_ms": 100.0, "total_ms": 200.0},
        {"utterance_type": "T1", "ttft_ms": None, "total_ms": None},
        {"utterance_type": "T2", "ttft_ms": 50.0, "total_ms": 150.0},
        {"utterance_type": "T2", "ttft_ms": None, "total_ms": 180.0},
    ]
    stats = latency_percentiles(rows)

    assert stats["T1"]["ttft_n_missing"] == 1
    assert stats["T1"]["total_n_missing"] == 1
    assert stats["T1"]["ttft_p50"] == 100.0
    assert stats["T1"]["total_p50"] == 200.0

    assert stats["T2"]["ttft_n_missing"] == 1
    assert stats["T2"]["total_n_missing"] == 0
    assert stats["T2"]["ttft_p50"] == 50.0
    assert stats["T2"]["total_p50"] == 150.0

    all_ref = stats["ALL(ref)"]
    assert all_ref["n"] == 4
    assert all_ref["ttft_n_missing"] == 2
    assert all_ref["total_n_missing"] == 1
    assert all_ref["ttft_p50"] == 50.0
    assert all_ref["total_p50"] == 180.0
