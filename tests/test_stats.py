"""Percentile math and index-size constraints."""

from __future__ import annotations

import pytest

from bench_lance.data import (
    choose_modulus,
    partitions_for,
    sample_rate_for,
    sub_vectors,
)
from bench_lance.stats import latency, percentile


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([4], 99) == 4


def test_latency_qps() -> None:
    report = latency([0.5, 1.5])
    assert report["serial_qps"] == pytest.approx(1.0)
    assert report["p50_ms"] == pytest.approx(1000)


def test_partition_training_fits() -> None:
    rows = 20_000
    parts = partitions_for(rows)
    assert parts * sample_rate_for(rows, parts) <= rows


def test_sub_vectors_divide_width() -> None:
    assert 128 % sub_vectors(128) == 0
    assert 32 % sub_vectors(32) == 0


def test_sample_rate_rejects_too_many_partitions() -> None:
    with pytest.raises(ValueError):
        sample_rate_for(10, 11)


def test_modulus_can_fill_k() -> None:
    assert choose_modulus(20_000, 10, 0) == 100
    assert choose_modulus(500, 10, 0) == 10
    with pytest.raises(ValueError):
        choose_modulus(50, 10, 100)
