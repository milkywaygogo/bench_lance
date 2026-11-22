"""Parquet to Lance and Lance to Parquet conversion throughput.

Each run is a child process. ``LANCE_CPU_THREADS`` sizes Lance's compute
pool once, when Lance starts, so a thread count can't change inside one
process. The child also sets Arrow's CPU pool to the same count.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence, TypeVar

import lance
import lance.fragment
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from bench_lance.data import mixed_batch, text_pool

VARIANTS = (
    "parquet_scan",
    "lance_scan",
    "p2l_single",
    "p2l_parallel",
    "l2p_single",
    "l2p_parallel",
)
"""Run kinds. ``single`` uses one writer. ``parallel`` uses one writer per
thread, each on its own slice of row groups or fragments."""

_READ_BATCH = 65_536
_SOURCE_PARQUET = "source.parquet"
_SOURCE_LANCE = "source.lance"
_SOURCE_META = "source.json"

T = TypeVar("T")


def split(items: Sequence[T], parts: int) -> list[list[T]]:
    """
    Cut a sequence into contiguous, nearly equal slices.

    Parameters
    ----------
    items : sequence
        Items to divide. Order is kept.
    parts : int
        Slices wanted. Must be positive. Fewer come back when there are
        fewer items than ``parts``.

    Returns
    -------
    list of list
        Non-empty slices whose concatenation is ``items``.

    Raises
    ------
    ValueError
        ``parts`` is not positive.

    Examples
    --------
    >>> split([1, 2, 3, 4, 5], 2)
    [[1, 2, 3], [4, 5]]
    >>> split([1, 2], 4)
    [[1], [2]]
    """
    if parts < 1:
        raise ValueError(f"parts must be positive, got {parts}")
    parts = min(parts, len(items))
    base, extra = divmod(len(items), parts) if parts else (0, 0)
    out = []
    start = 0
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        out.append(list(items[start:end]))
        start = end
    return out


def write_source(path: Path, rows: int, dim: int, seed: int, group_rows: int) -> int:
    """
    Write the synthetic table as one Parquet file.

    Parameters
    ----------
    path : pathlib.Path
        Output file.
    rows : int
        Table rows. Must be positive.
    dim : int
        Width of the ``vector`` column.
    seed : int
        Generator seed.
    group_rows : int
        Rows per row group. Each group is one generated batch.

    Returns
    -------
    int
        In-memory Arrow bytes of the whole table.
    """
    if rows < 1:
        raise ValueError(f"rows must be positive, got {rows}")
    pool = text_pool(seed)
    arrow_bytes = 0
    writer = None
    try:
        for start in range(0, rows, group_rows):
            batch = mixed_batch(start, min(group_rows, rows - start), dim, seed, pool)
            if writer is None:
                writer = pq.ParquetWriter(path, batch.schema)
            writer.write_batch(batch, row_group_size=group_rows)
            arrow_bytes += batch.nbytes
    finally:
        if writer is not None:
            writer.close()
    return arrow_bytes


def parquet_to_lance(src: Path, dst: Path, workers: int, group_rows: int) -> None:
    """
    Convert one Parquet file into a Lance dataset.

    One worker streams the file through ``lance.write_dataset``. More
    workers each take a contiguous run of row groups, write Lance
    fragments with ``lance.fragment.write_fragments``, and the fragments
    are committed together in row order.

    Parameters
    ----------
    src : pathlib.Path
        Parquet file.
    dst : pathlib.Path
        Lance dataset to create or overwrite.
    workers : int
        Concurrent writers. Capped at the number of row groups.
    group_rows : int
        Maximum rows per Lance fragment.
    """
    dataset = pds.dataset(src, format="parquet")
    schema = dataset.schema
    if workers == 1:
        reader = dataset.scanner(batch_size=_READ_BATCH).to_reader()
        lance.write_dataset(
            reader, dst, mode="overwrite", max_rows_per_file=group_rows
        )
        return
    (fragment,) = dataset.get_fragments()
    pieces = [
        fragment.subset(row_group_ids=ids)
        for ids in split(range(fragment.num_row_groups), workers)
    ]

    def write(piece: pds.ParquetFileFragment) -> list:
        reader = piece.scanner(schema=schema, batch_size=_READ_BATCH).to_reader()
        return lance.fragment.write_fragments(
            reader, dst, schema=schema, max_rows_per_file=group_rows
        )

    with ThreadPoolExecutor(len(pieces)) as pool:
        written = [meta for part in pool.map(write, pieces) for meta in part]
    lance.LanceDataset.commit(dst, lance.LanceOperation.Overwrite(schema, written))


def lance_to_parquet(src: Path, dst: Path, workers: int, group_rows: int) -> list[Path]:
    """
    Convert a Lance dataset into Parquet files.

    Each worker scans a contiguous run of fragments and writes one file,
    ``part-NNNN.parquet``. One worker writes a single file.

    Parameters
    ----------
    src : pathlib.Path
        Lance dataset.
    dst : pathlib.Path
        Output directory. Created if missing.
    workers : int
        Concurrent writers. Capped at the number of fragments.
    group_rows : int
        Rows per Lance read batch, and so per Parquet row group.

    Returns
    -------
    list of pathlib.Path
        Written files, in row order.
    """
    dst.mkdir(parents=True, exist_ok=True)
    dataset = lance.dataset(src)
    groups = split(dataset.get_fragments(), workers)
    paths = [dst / f"part-{i:04d}.parquet" for i in range(len(groups))]

    def write(index: int) -> None:
        reader = dataset.scanner(
            fragments=groups[index], batch_size=group_rows, scan_in_order=True
        ).to_reader()
        with pq.ParquetWriter(paths[index], reader.schema) as writer:
            for batch in reader:
                writer.write_batch(batch)

    with ThreadPoolExecutor(len(groups)) as pool:
        list(pool.map(write, range(len(groups))))
    return paths


def _count_batches(reader: pa.RecordBatchReader) -> int:
    return sum(batch.num_rows for batch in reader)


def _dir_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _peak_rss_bytes() -> int:
    # ru_maxrss survives exec, so a child would report the parent's peak.
    # VmHWM belongs to this process's own address space.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM missing from /proc/self/status")


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _run_once(cfg: dict) -> dict:
    """Time one variant in this process. ``cfg`` comes from the parent."""
    variant = cfg["variant"]
    workers = cfg["threads"] if variant.endswith("parallel") else 1
    workdir = Path(cfg["workdir"])
    src_parquet = workdir / _SOURCE_PARQUET
    src_lance = workdir / _SOURCE_LANCE
    out = workdir / "out"
    group_rows = cfg["group_rows"]
    pa.set_cpu_count(cfg["threads"])
    shutil.rmtree(out, ignore_errors=True)
    os.sync()

    cpu0 = _cpu_seconds()
    start = time.perf_counter()
    if variant == "parquet_scan":
        reader = pds.dataset(src_parquet).scanner(batch_size=_READ_BATCH).to_reader()
        rows = _count_batches(reader)
    elif variant == "lance_scan":
        reader = lance.dataset(src_lance).scanner(batch_size=_READ_BATCH).to_reader()
        rows = _count_batches(reader)
    elif variant.startswith("p2l"):
        parquet_to_lance(src_parquet, out, workers, group_rows)
    elif variant.startswith("l2p"):
        lance_to_parquet(src_lance, out, workers, group_rows)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    wall = time.perf_counter() - start
    cpu = _cpu_seconds() - cpu0

    start = time.perf_counter()
    os.sync()
    sync = time.perf_counter() - start

    out_bytes = 0
    if variant.startswith("p2l"):
        rows = lance.dataset(out).count_rows()
        out_bytes = _dir_bytes(out)
    elif variant.startswith("l2p"):
        files = sorted(out.glob("*.parquet"))
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        out_bytes = _dir_bytes(out)
    shutil.rmtree(out, ignore_errors=True)
    return {
        "variant": variant,
        "threads": cfg["threads"],
        "workers": workers,
        "rows": rows,
        "wall_s": wall,
        "sync_s": sync,
        "cpu_s": cpu,
        "peak_rss_bytes": _peak_rss_bytes(),
        "out_bytes": out_bytes,
    }


def _spawn(cfg: dict) -> dict:
    env = dict(os.environ)
    env["LANCE_CPU_THREADS"] = str(cfg["threads"])
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (package_root, env.get("PYTHONPATH")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-m", "bench_lance.convert_bench", json.dumps(cfg)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cfg['variant']} threads={cfg['threads']} failed "
            f"(exit {proc.returncode}):\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _prepare(rows: int, dim: int, seed: int, group_rows: int, workdir: Path) -> dict:
    """Write the Parquet and Lance sources, or reuse matching ones."""
    key = {"rows": rows, "dim": dim, "seed": seed, "group_rows": group_rows}
    meta_path = workdir / _SOURCE_META
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if {k: meta.get(k) for k in key} == key:
            return meta
    shutil.rmtree(workdir / _SOURCE_LANCE, ignore_errors=True)
    start = time.perf_counter()
    arrow_bytes = write_source(workdir / _SOURCE_PARQUET, rows, dim, seed, group_rows)
    gen_s = time.perf_counter() - start
    parquet_to_lance(
        workdir / _SOURCE_PARQUET,
        workdir / _SOURCE_LANCE,
        os.cpu_count() or 1,
        group_rows,
    )
    dataset = lance.dataset(workdir / _SOURCE_LANCE)
    meta = {
        **key,
        "arrow_bytes": arrow_bytes,
        "parquet_bytes": _dir_bytes(workdir / _SOURCE_PARQUET),
        "lance_bytes": _dir_bytes(workdir / _SOURCE_LANCE),
        "lance_fragments": len(dataset.get_fragments()),
        "lance_storage_version": dataset.data_storage_version,
        "generate_s": gen_s,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def run_convert(
    rows: int,
    dim: int,
    threads: Sequence[int],
    seed: int,
    workdir: Path,
    group_rows: int = 1 << 20,
    variants: Sequence[str] = VARIANTS,
) -> dict:
    """
    Time Parquet-to-Lance and Lance-to-Parquet conversion at each thread count.

    The sources are written once and reused while ``rows``, ``dim``,
    ``seed``, and ``group_rows`` match. Every run reads warm sources from
    the page cache. ``wall_s`` ends when the writer returns. ``sync_s`` is
    the following ``os.sync``, which flushes the written bytes to disk.

    Parameters
    ----------
    rows : int
        Table rows.
    dim : int
        Width of the ``vector`` column.
    threads : sequence of int
        Thread counts to sweep. Each must be positive.
    seed : int
        Generator seed.
    workdir : pathlib.Path
        Directory for the sources and each run's output.
    group_rows : int
        Rows per Parquet row group, Lance fragment, and Lance read batch.
    variants : sequence of str
        Runs to make, from :data:`VARIANTS`.

    Returns
    -------
    dict
        Source sizes and one entry per variant and thread count, with
        wall, sync, and CPU seconds, peak RSS, rows, and output bytes.

    Raises
    ------
    ValueError
        A thread count is not positive or a variant is unknown.
    RuntimeError
        A run failed or wrote the wrong number of rows.
    """
    if any(t < 1 for t in threads):
        raise ValueError(f"thread counts must be positive, got {list(threads)}")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; pick from {VARIANTS}")
    workdir.mkdir(parents=True, exist_ok=True)
    source = _prepare(rows, dim, seed, group_rows, workdir)
    runs = []
    for variant in variants:
        for count in threads:
            cfg = {
                "variant": variant,
                "threads": count,
                "workdir": str(workdir),
                "group_rows": group_rows,
            }
            run = _spawn(cfg)
            if run["rows"] != rows:
                raise RuntimeError(
                    f"{variant} threads={count} produced {run['rows']} rows, "
                    f"expected {rows}"
                )
            runs.append(run)
    return {
        "kind": "convert",
        "cpus": os.cpu_count(),
        "lance_version": lance.__version__,
        "pyarrow_version": pa.__version__,
        "source": source,
        "runs": runs,
    }


if __name__ == "__main__":
    print(json.dumps(_run_once(json.loads(sys.argv[1]))))
