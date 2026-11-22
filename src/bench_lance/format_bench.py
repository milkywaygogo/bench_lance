"""Lance file format against Parquet: write, scan, filter, and random take."""

from __future__ import annotations

import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from bench_lance.data import make_table
from bench_lance.stats import median

_FILTER_MOD = 100
_LANCE_FILTER = f"id % {_FILTER_MOD} == 0"


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _time(fn, rounds: int) -> float:
    fn()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return median(samples)


def run_format(
    rows: int,
    dim: int,
    take: int,
    rounds: int,
    seed: int,
    workdir: Path,
) -> dict:
    """
    Write one table both ways and time warm-cache reads.

    Parquet take decodes the file, then slices rows. Lance take reads the
    requested rows. Both runs follow one discarded call, so the page cache
    is warm.

    Parameters
    ----------
    rows : int
        Table rows.
    dim : int
        Vector width.
    take : int
        Row ids fetched per take. Must be in ``1 .. rows``.
    rounds : int
        Timed repetitions. The median is reported. One discarded call
        precedes them.
    seed : int
        Generator seed for the table and the take ids.
    workdir : pathlib.Path
        Directory for ``data.lance`` and ``data.parquet``.

    Returns
    -------
    dict
        Write seconds, byte sizes, and median scan, filter, and take
        seconds for Lance and Parquet. ``cache`` is ``warm``.

    Raises
    ------
    ValueError
        ``take`` is outside ``1 .. rows``.
    """
    if take < 1 or take > rows:
        raise ValueError(f"take must be in 1..{rows}, got {take}")
    workdir.mkdir(parents=True, exist_ok=True)
    table, _vecs = make_table(rows, dim, seed, modulus=_FILTER_MOD)
    lance_path = workdir / "data.lance"
    parquet_path = workdir / "data.parquet"

    start = time.perf_counter()
    lance.write_dataset(table, lance_path, mode="overwrite")
    lance_write_s = time.perf_counter() - start

    start = time.perf_counter()
    pq.write_table(table, parquet_path)
    parquet_write_s = time.perf_counter() - start

    dataset = lance.dataset(lance_path)
    rng_ids = _take_ids(rows, take, seed)

    def lance_scan() -> pa.Table:
        return dataset.scanner().to_table()

    def parquet_scan() -> pa.Table:
        return pq.read_table(parquet_path)

    def lance_filter() -> pa.Table:
        return dataset.scanner(columns=["id"], filter=_LANCE_FILTER).to_table()

    arrow_filter = pc.equal(pc.modulo(pds.field("id"), _FILTER_MOD), 0)
    parquet_dataset = pds.dataset(parquet_path, format="parquet")

    def parquet_filter() -> pa.Table:
        return parquet_dataset.to_table(columns=["id"], filter=arrow_filter)

    def lance_take() -> pa.Table:
        return dataset.take(rng_ids)

    def parquet_take() -> pa.Table:
        return pq.read_table(parquet_path).take(rng_ids)

    return {
        "kind": "format",
        "rows": rows,
        "dim": dim,
        "take_ids": take,
        "rounds": rounds,
        "cache": "warm",
        "lance_write_s": lance_write_s,
        "parquet_write_s": parquet_write_s,
        "lance_bytes": _dir_bytes(lance_path),
        "parquet_bytes": parquet_path.stat().st_size,
        "lance_scan_s": _time(lance_scan, rounds),
        "parquet_scan_s": _time(parquet_scan, rounds),
        "lance_filter_s": _time(lance_filter, rounds),
        "parquet_filter_s": _time(parquet_filter, rounds),
        "lance_take_s": _time(lance_take, rounds),
        "parquet_take_s": _time(parquet_take, rounds),
    }


def _take_ids(rows: int, take: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 1).integers(0, rows, size=take)
