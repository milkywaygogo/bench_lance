"""Command line for the Lance and LanceDB benchmarks."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from bench_lance.convert_bench import VARIANTS, run_convert
from bench_lance.format_bench import run_format
from bench_lance.vector_bench import run_vector


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, help="Write the JSON report to this path")
    parser.add_argument(
        "--workdir", type=Path, help="Directory for datasets. Defaults to a temp dir"
    )
    parser.add_argument("--seed", type=int, default=0)


def _add_size(parser: argparse.ArgumentParser, rows: int) -> None:
    parser.add_argument("--rows", type=int, default=rows)
    parser.add_argument("--dim", type=int, default=128)


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--take", type=int, default=1000, help="Row ids fetched per take"
    )
    parser.add_argument("--rounds", type=int, default=5)


def _add_vector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--index", choices=["ivf_hnsw_sq", "ivf_pq"], default="ivf_hnsw_sq"
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=0,
        help="IVF partitions. 0 picks from the row count",
    )
    parser.add_argument("--nprobes", default="1,4,16")
    parser.add_argument("--refine", default="1,5", help="refine_factor values")
    parser.add_argument(
        "--filter-mod",
        type=int,
        default=0,
        help=(
            "Filter label = id %% N. 0 uses 100 when the table is large enough, else 10"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="bench_lance", description=__doc__)
    commands = root.add_subparsers(dest="cmd", required=True)

    fmt = commands.add_parser("format", help="Lance format against Parquet")
    _add_shared(fmt)
    _add_size(fmt, 100_000)
    _add_format(fmt)

    vec = commands.add_parser("vector", help="LanceDB ingest, index, and search")
    _add_shared(vec)
    _add_size(vec, 20_000)
    _add_vector(vec)

    conv = commands.add_parser(
        "convert", help="Parquet to Lance and Lance to Parquet throughput"
    )
    _add_shared(conv)
    conv.add_argument("--rows", type=int, default=16 * 1024 * 1024)
    conv.add_argument("--dim", type=int, default=32)
    conv.add_argument(
        "--threads", default="1,2,4,8,16,32,64", help="Thread counts to sweep"
    )
    conv.add_argument(
        "--group-rows",
        type=int,
        default=1 << 20,
        help="Rows per Parquet row group and Lance fragment",
    )
    conv.add_argument(
        "--variants", default=",".join(VARIANTS), help="Comma-separated runs"
    )

    both = commands.add_parser(
        "all", help="Run the format benchmark, then the vector benchmark"
    )
    _add_shared(both)
    _add_size(both, 20_000)
    _add_format(both)
    _add_vector(both)
    return root


def _workdir(path: Path | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        return path, None
    temp = tempfile.TemporaryDirectory(prefix="bench-lance-")
    return Path(temp.name), temp


def _mb(nbytes: int) -> str:
    return f"{nbytes / 1e6:.1f} MB"


def _print_format(report: dict) -> None:
    print(
        f"format  rows={report['rows']} dim={report['dim']} "
        f"take={report['take_ids']} rounds={report['rounds']} cache={report['cache']}"
    )
    print(
        f"  write    lance {report['lance_write_s']:.3f}s"
        f"    parquet {report['parquet_write_s']:.3f}s"
    )
    print(
        f"  bytes    lance {_mb(report['lance_bytes'])}"
        f"    parquet {_mb(report['parquet_bytes'])}"
    )
    print(
        f"  scan     lance {report['lance_scan_s'] * 1e3:.1f} ms"
        f"    parquet {report['parquet_scan_s'] * 1e3:.1f} ms"
    )
    print(
        f"  filter   lance {report['lance_filter_s'] * 1e3:.1f} ms"
        f"    parquet {report['parquet_filter_s'] * 1e3:.1f} ms"
    )
    print(
        f"  take     lance {report['lance_take_s'] * 1e3:.1f} ms"
        f"    parquet {report['parquet_take_s'] * 1e3:.1f} ms"
        "   (parquet decodes the file)"
    )


def _print_point(point: dict) -> None:
    print(
        f"  {point['nprobes']:>7}  {point['refine_factor']:>6}"
        f"  {point['recall_at_k']:>6.3f}  {point['p50_ms']:>8.2f}"
        f"  {point['p99_ms']:>8.2f}  {point['serial_qps']:>8.1f}"
    )


def _print_vector(report: dict) -> None:
    index = report["index"]
    print(
        f"vector  rows={report['rows']} dim={report['dim']} "
        f"queries={report['queries']} "
        f"k={report['k']} metric={report['metric']} cache={report['cache']}"
    )
    print(
        f"  ingest {report['ingest_s']:.3f}s    "
        f"index {index['type']} partitions={index['num_partitions']} "
        f"in {index['build_s']:.3f}s    {_mb(report['bytes'])}"
    )
    flat = report["flat"]
    print(
        f"  flat exact   p50 {flat['p50_ms']:.2f} ms"
        f"   p99 {flat['p99_ms']:.2f} ms   serial qps {flat['serial_qps']:.1f}"
    )
    print("  nprobes  refine  recall     p50       p99   serial_qps")
    for point in report["points"]:
        _print_point(point)
    filt = report["filter"]
    print(
        f"  filter {filt['where']}  selectivity {filt['selectivity']:.3f}"
        f"  nprobes {filt['nprobes']}  refine {filt['refine_factor']}"
        f"  recall {filt['recall_at_k']:.3f}"
        f"  p50 {filt['p50_ms']:.2f} ms  p99 {filt['p99_ms']:.2f} ms"
    )


def _print_convert(report: dict) -> None:
    src = report["source"]
    print(
        f"convert  rows={src['rows']} dim={src['dim']} cpus={report['cpus']} "
        f"lance={report['lance_version']} pyarrow={report['pyarrow_version']}"
    )
    print(
        f"  source   arrow {_mb(src['arrow_bytes'])}"
        f"    parquet {_mb(src['parquet_bytes'])}"
        f"    lance {_mb(src['lance_bytes'])} ({src['lance_fragments']} fragments,"
        f" format {src['lance_storage_version']})"
    )
    print(
        "  variant        threads  wall_s  sync_s  Mrows/s  arrow_GB/s"
        "  cores  rss_GB  out_GB"
    )
    for run in report["runs"]:
        wall = run["wall_s"]
        print(
            f"  {run['variant']:<14} {run['threads']:>7} {wall:>7.2f}"
            f" {run['sync_s']:>7.2f} {run['rows'] / wall / 1e6:>8.2f}"
            f" {src['arrow_bytes'] / wall / 1e9:>11.2f}"
            f" {run['cpu_s'] / wall:>6.1f} {run['peak_rss_bytes'] / 1e9:>7.1f}"
            f" {run['out_bytes'] / 1e9:>7.2f}"
        )


def _emit(report: dict, out: Path | None) -> None:
    if report.get("kind") == "convert":
        _print_convert(report)
    if report.get("format"):
        _print_format(report["format"])
    elif report.get("kind") == "format":
        _print_format(report)
    if report.get("vector"):
        _print_vector(report["vector"])
    elif report.get("kind") == "vector":
        _print_vector(report)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    """
    Run one benchmark command.

    Parameters
    ----------
    argv : list of str or None
        Arguments after the program name. ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process status. ``0`` means the report was printed.
    """
    args = _parser().parse_args(argv)
    workdir, temp = _workdir(args.workdir)
    try:
        if args.cmd == "format":
            report = run_format(
                args.rows,
                args.dim,
                args.take,
                args.rounds,
                args.seed,
                workdir / "format",
            )
        elif args.cmd == "convert":
            report = run_convert(
                args.rows,
                args.dim,
                [int(t) for t in args.threads.split(",")],
                args.seed,
                workdir / "convert",
                args.group_rows,
                args.variants.split(","),
            )
        elif args.cmd == "vector":
            report = run_vector(
                args.rows,
                args.dim,
                args.queries,
                args.k,
                args.index,
                args.partitions,
                args.nprobes,
                args.refine,
                args.filter_mod,
                args.seed,
                workdir / "vector",
            )
        else:
            report = {
                "format": run_format(
                    args.rows,
                    args.dim,
                    args.take,
                    args.rounds,
                    args.seed,
                    workdir / "format",
                ),
                "vector": run_vector(
                    args.rows,
                    args.dim,
                    args.queries,
                    args.k,
                    args.index,
                    args.partitions,
                    args.nprobes,
                    args.refine,
                    args.filter_mod,
                    args.seed,
                    workdir / "vector",
                ),
            }
    finally:
        if temp is not None:
            temp.cleanup()
    _emit(report, args.out)
    return 0
