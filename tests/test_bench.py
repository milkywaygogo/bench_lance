"""Small end-to-end runs of the format and vector benchmarks."""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bench_lance.convert_bench import (
    VARIANTS,
    lance_to_parquet,
    parquet_to_lance,
    run_convert,
    write_source,
)
from bench_lance.format_bench import run_format
from bench_lance.vector_bench import run_vector


def test_format_small(tmp_path: Path) -> None:
    report = run_format(rows=2_000, dim=16, take=32, rounds=2, seed=1, workdir=tmp_path)
    assert report["lance_bytes"] > 0
    assert report["parquet_bytes"] > 0
    assert report["lance_take_s"] > 0
    assert report["parquet_scan_s"] > 0


def test_vector_small(tmp_path: Path) -> None:
    report = run_vector(
        rows=1_024,
        dim=32,
        queries=5,
        k=10,
        index="ivf_hnsw_sq",
        partitions=1,
        nprobes="1",
        refine="1",
        filter_mod=10,
        seed=2,
        workdir=tmp_path,
    )
    assert report["index"]["build_s"] > 0
    assert report["points"][0]["nprobes"] == 1
    assert 0 <= report["points"][0]["recall_at_k"] <= 1
    assert 0 <= report["filter"]["recall_at_k"] <= 1


@pytest.mark.parametrize("workers", [1, 3])
def test_convert_round_trip(tmp_path: Path, workers: int) -> None:
    rows, group_rows = 10_000, 1_024
    source = tmp_path / "src.parquet"
    write_source(source, rows, dim=4, seed=3, group_rows=group_rows)
    parquet_to_lance(source, tmp_path / "t.lance", workers, group_rows)
    files = lance_to_parquet(
        tmp_path / "t.lance", tmp_path / "back", workers, group_rows
    )
    assert len(files) == workers
    expected = pq.read_table(source)
    assert lance.dataset(tmp_path / "t.lance").to_table().equals(expected)
    assert pa.concat_tables(pq.read_table(f) for f in files).equals(expected)


def test_convert_small(tmp_path: Path) -> None:
    report = run_convert(
        rows=5_000,
        dim=4,
        threads=[1, 2],
        seed=1,
        workdir=tmp_path,
        group_rows=1_024,
    )
    assert len(report["runs"]) == len(VARIANTS) * 2
    assert all(run["rows"] == 5_000 for run in report["runs"])
    assert report["source"]["lance_fragments"] == 5
