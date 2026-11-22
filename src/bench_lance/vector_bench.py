"""LanceDB ingest, index build, and serial search at a measured recall."""

from __future__ import annotations

import time
from pathlib import Path

import lancedb
import numpy as np
from lancedb.index import HnswSq, IvfPq

from bench_lance.data import (
    choose_modulus,
    make_table,
    partitions_for,
    sample_rate_for,
    sub_vectors,
    unit_vectors,
)
from bench_lance.stats import latency

# Enough queries to fault the index into the page cache before timing.
_WARMUP_QUERIES = 10
# Small HNSW graph. Larger values raise recall and build time.
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 150


def _config(kind: str, dim: int, partitions: int, sample_rate: int):
    if kind == "ivf_hnsw_sq":
        return HnswSq(
            distance_type="cosine",
            num_partitions=partitions,
            sample_rate=sample_rate,
            m=_HNSW_M,
            ef_construction=_HNSW_EF_CONSTRUCTION,
        )
    if kind == "ivf_pq":
        return IvfPq(
            distance_type="cosine",
            num_partitions=partitions,
            num_sub_vectors=sub_vectors(dim),
            sample_rate=sample_rate,
        )
    raise ValueError(f"unknown index {kind}")


def _ids(rows: list[dict]) -> list[int]:
    return [int(row["id"]) for row in rows]


def _recall(got: list[int], truth: list[int], k: int) -> float:
    return len(set(got) & set(truth)) / k


def _search(
    table,
    query: np.ndarray,
    k: int,
    *,
    where: str | None,
    nprobes: int | None,
    refine: int | None,
):
    builder = table.search(query).select(["id", "_distance"]).limit(k)
    if where is not None:
        builder = builder.where(where, prefilter=True)
    if nprobes is None:
        builder = builder.bypass_vector_index()
    else:
        factor = 1 if refine is None else refine
        builder = builder.nprobes(nprobes).refine_factor(factor)
    return _ids(builder.to_list())


def _time_queries(fn, queries: np.ndarray) -> tuple[list[float], list[list[int]]]:
    samples = []
    found = []
    for query in queries:
        start = time.perf_counter()
        ids = fn(query)
        samples.append(time.perf_counter() - start)
        found.append(ids)
    return samples, found


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parse_ints(text: str, name: str) -> list[int]:
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise ValueError(f"{name} values must be >= 1, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{name} is empty, got {text!r}")
    return values


def _sweep(
    table, queries: np.ndarray, k: int, nprobes: int, refine: int, where: str | None
):
    warmup = queries[: min(_WARMUP_QUERIES, len(queries))]
    _time_queries(
        lambda query: _search(
            table, query, k, where=where, nprobes=nprobes, refine=refine
        ),
        warmup,
    )
    return _time_queries(
        lambda query: _search(
            table, query, k, where=where, nprobes=nprobes, refine=refine
        ),
        queries,
    )


def run_vector(
    rows: int,
    dim: int,
    queries: int,
    k: int,
    index: str,
    partitions: int,
    nprobes: str,
    refine: str,
    filter_mod: int,
    seed: int,
    workdir: Path,
) -> dict:
    """
    Load a table, build one index, and sweep search parameters.

    Ground truth is an exact search (``bypass_vector_index``) on the same
    table. Recall is the mean of ``|approx ∩ exact| / k``. Latency is one
    serial client returning ids only. A short warmup runs first so the
    page cache is warm. Query vectors are not inserted.

    Parameters
    ----------
    rows : int
        Inserted vectors.
    dim : int
        Vector width.
    queries : int
        Held-out query count. Must be positive.
    k : int
        Neighbors. Must be in ``1 .. rows``.
    index : str
        ``ivf_hnsw_sq`` or ``ivf_pq``. Distance is cosine.
    partitions : int
        IVF partitions. ``0`` picks a count from ``rows``.
    nprobes : str
        Comma-separated probe counts. Each is capped at the partition count.
    refine : str
        Comma-separated ``refine_factor`` values.
    filter_mod : int
        ``label = 0`` uses ``id % filter_mod``. ``0`` picks the modulus.
    seed : int
        Generator seed. Queries use ``seed + 1``.
    workdir : pathlib.Path
        Directory for the LanceDB database.

    Returns
    -------
    dict
        Ingest seconds, index build, flat exact latency, one point per
        probe and refine pair, and one filtered point.

    Raises
    ------
    ValueError
        ``queries``, ``k``, ``index``, ``partitions``, ``nprobes``, or
        ``refine`` is not usable.
    """
    if queries < 1:
        raise ValueError(f"queries must be positive, got {queries}")
    if k < 1 or k > rows:
        raise ValueError(f"k must be in 1..{rows}, got {k}")
    modulus = choose_modulus(rows, k, filter_mod)
    npart = partitions if partitions else partitions_for(rows)
    if npart > rows:
        raise ValueError(f"{npart} partitions is more than {rows} rows")
    rate = sample_rate_for(rows, npart)
    probe_values = [min(value, npart) for value in _parse_ints(nprobes, "nprobes")]
    refine_values = _parse_ints(refine, "refine")

    workdir.mkdir(parents=True, exist_ok=True)
    table_data, _inserted = make_table(rows, dim, seed, modulus)
    query_vecs = unit_vectors(queries, dim, seed + 1)
    where = "label = 0"

    db = lancedb.connect(workdir / "db")
    start = time.perf_counter()
    table = db.create_table("items", data=table_data, mode="overwrite")
    ingest_s = time.perf_counter() - start

    flat_samples, truth = _time_queries(
        lambda query: _search(table, query, k, where=None, nprobes=None, refine=None),
        query_vecs,
    )
    filter_truth = [
        _search(table, query, k, where=where, nprobes=None, refine=None)
        for query in query_vecs
    ]

    config = _config(index, dim, npart, rate)
    start = time.perf_counter()
    table.create_index("vector", config=config)
    index_s = time.perf_counter() - start

    points = []
    for refine_factor in refine_values:
        for nprobe in probe_values:
            samples, found = _sweep(table, query_vecs, k, nprobe, refine_factor, None)
            recalls = [_recall(got, exact, k) for got, exact in zip(found, truth)]
            points.append(
                {
                    "nprobes": nprobe,
                    "refine_factor": refine_factor,
                    "recall_at_k": sum(recalls) / len(recalls),
                    **latency(samples),
                }
            )

    filter_nprobes = max(probe_values)
    filter_refine = refine_values[0]
    filter_samples, filter_found = _sweep(
        table, query_vecs, k, filter_nprobes, filter_refine, where
    )
    filter_recalls = [
        _recall(got, exact, k) for got, exact in zip(filter_found, filter_truth)
    ]

    return {
        "kind": "vector",
        "rows": rows,
        "dim": dim,
        "queries": queries,
        "k": k,
        "metric": "cosine",
        "cache": "warm",
        "ingest_s": ingest_s,
        "bytes": _dir_bytes(workdir / "db"),
        "index": {
            "type": index,
            "num_partitions": npart,
            "sample_rate": rate,
            "num_sub_vectors": sub_vectors(dim) if index == "ivf_pq" else None,
            "m": _HNSW_M if index == "ivf_hnsw_sq" else None,
            "ef_construction": _HNSW_EF_CONSTRUCTION
            if index == "ivf_hnsw_sq"
            else None,
            "build_s": index_s,
        },
        "flat": latency(flat_samples),
        "points": points,
        "filter": {
            "where": where,
            "modulus": modulus,
            "selectivity": 1 / modulus,
            "nprobes": filter_nprobes,
            "refine_factor": filter_refine,
            "recall_at_k": sum(filter_recalls) / len(filter_recalls),
            **latency(filter_samples),
        },
    }
