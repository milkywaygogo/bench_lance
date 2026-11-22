"""Latency summaries for one serial client."""

from __future__ import annotations


def percentile(samples: list[float], pct: float) -> float:
    """
    Linear percentile of a sample.

    Parameters
    ----------
    samples : list of float
        Observations. Must be non-empty.
    pct : float
        Percentile in the closed range 0 to 100.

    Returns
    -------
    float
        Interpolated value. One observation returns that observation.

    Raises
    ------
    ValueError
        ``samples`` is empty or ``pct`` is outside 0 to 100.

    Examples
    --------
    >>> percentile([1, 2, 3, 4], 50)
    2.5
    >>> percentile([4], 99)
    4
    """
    if not samples:
        raise ValueError(f"no samples for percentile {pct}")
    if not 0 <= pct <= 100:
        raise ValueError(f"percentile out of range: {pct}")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def latency(samples_s: list[float]) -> dict[str, float]:
    """
    Summarize serial-client latencies.

    Parameters
    ----------
    samples_s : list of float
        Elapsed seconds for each call. Must be non-empty, and the total
        must be positive.

    Returns
    -------
    dict of str to float
        ``p50_ms``, ``p99_ms``, ``mean_ms``, and ``serial_qps``.
        The rate is the number of calls divided by the total seconds.

    Raises
    ------
    ValueError
        ``samples_s`` is empty.

    Examples
    --------
    >>> report = latency([0.5, 1.5])
    >>> report["serial_qps"]
    1.0
    >>> report["p50_ms"]
    1000.0
    """
    total = sum(samples_s)
    return {
        "p50_ms": percentile(samples_s, 50) * 1e3,
        "p99_ms": percentile(samples_s, 99) * 1e3,
        "mean_ms": (total / len(samples_s)) * 1e3,
        "serial_qps": len(samples_s) / total,
    }


def median(samples: list[float]) -> float:
    """
    Linear percentile at 50.

    Parameters
    ----------
    samples : list of float
        Observations. Must be non-empty.

    Returns
    -------
    float
        Interpolated median.

    Raises
    ------
    ValueError
        ``samples`` is empty.

    Examples
    --------
    >>> median([1, 2, 3, 4])
    2.5
    """
    return percentile(samples, 50)
