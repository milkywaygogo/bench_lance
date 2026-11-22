"""Synthetic tables: unit vectors for search, mixed columns for conversion."""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def sub_vectors(dim: int) -> int:
    """
    Product-quantization width that divides ``dim``.

    Parameters
    ----------
    dim : int
        Vector width. Must be positive. Prefer ``dim / 8``.

    Returns
    -------
    int
        Largest divisor of ``dim`` that is at most ``dim / 8``.

    Raises
    ------
    ValueError
        ``dim`` is not positive.

    Examples
    --------
    >>> sub_vectors(128)
    16
    >>> sub_vectors(32)
    4
    """
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    target = max(1, dim // 8)
    for candidate in range(target, 0, -1):
        if dim % candidate == 0:
            return candidate
    raise ValueError(f"no product-quantization width divides dim {dim}")


def partitions_for(rows: int) -> int:
    """
    IVF partition count for a table.

    Parameters
    ----------
    rows : int
        Table rows. Must be positive. At least 256 rows per partition,
        and at most 256 partitions.

    Returns
    -------
    int
        Partition count.

    Raises
    ------
    ValueError
        ``rows`` is not positive.

    Examples
    --------
    >>> partitions_for(20_000)
    78
    """
    if rows < 1:
        raise ValueError(f"rows must be positive, got {rows}")
    return max(1, min(256, rows // 256))


def sample_rate_for(rows: int, partitions: int) -> int:
    """
    K-means sample rate that fits in the table.

    LanceDB trains on ``sample_rate * partitions`` vectors. The rate is
    capped at 256.

    Parameters
    ----------
    rows : int
        Table rows.
    partitions : int
        IVF partitions. Must be positive and no larger than ``rows``.

    Returns
    -------
    int
        Sample rate.

    Raises
    ------
    ValueError
        ``partitions`` is not positive or exceeds ``rows``.

    Examples
    --------
    >>> sample_rate_for(20_000, 78)
    256
    >>> sample_rate_for(10, 11)
    Traceback (most recent call last):
    ...
    ValueError: 11 partitions is more than 10 rows
    """
    if partitions < 1:
        raise ValueError(f"partitions must be positive, got {partitions}")
    if rows < partitions:
        raise ValueError(f"{partitions} partitions is more than {rows} rows")
    return max(1, min(256, rows // partitions))


def choose_modulus(rows: int, k: int, requested: int) -> int:
    """
    Label modulus whose matches can fill ``k`` neighbors.

    Parameters
    ----------
    rows : int
        Table rows.
    k : int
        Neighbors requested.
    requested : int
        Modulus to use. ``0`` selects 100 when ``rows`` can fill ``k`` at
        that selectivity, and 10 otherwise.

    Returns
    -------
    int
        Modulus. ``id % modulus == 0`` matches at least ``k`` rows.

    Raises
    ------
    ValueError
        The modulus is not positive, or it matches fewer than ``k`` rows.

    Examples
    --------
    >>> choose_modulus(20_000, 10, 0)
    100
    >>> choose_modulus(500, 10, 0)
    10
    """
    modulus = requested if requested else (100 if rows >= 100 * k else 10)
    if modulus < 1:
        raise ValueError(f"filter modulus must be positive, got {modulus}")
    if rows // modulus < k:
        raise ValueError(
            f"filter id % {modulus} matches {rows // modulus} rows, "
            f"which is under k={k}"
        )
    return modulus


def unit_vectors(rows: int, dim: int, seed: int) -> np.ndarray:
    """
    Draw L2-normalized rows.

    Parameters
    ----------
    rows : int
        Number of vectors.
    dim : int
        Vector width.
    seed : int
        Generator seed.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(rows, dim)``. Each row has norm 1.

    Examples
    --------
    >>> vecs = unit_vectors(2, 4, 0)
    >>> vecs.shape
    (2, 4)
    >>> bool(np.allclose(np.linalg.norm(vecs, axis=1), 1))
    True
    """
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((rows, dim), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs /= norms
    return vecs


def make_table(
    rows: int, dim: int, seed: int, modulus: int
) -> tuple[pa.Table, np.ndarray]:
    """
    Build an Arrow table and the vector matrix it stores.

    Parameters
    ----------
    rows : int
        Table rows.
    dim : int
        Vector width.
    seed : int
        Generator seed.
    modulus : int
        ``label`` is ``id % modulus``.

    Returns
    -------
    tuple of pyarrow.Table and numpy.ndarray
        Columns ``id``, ``label``, and unit ``vector``, plus the matrix.

    Examples
    --------
    >>> table, vecs = make_table(4, 4, 0, 2)
    >>> table.column_names
    ['id', 'label', 'vector']
    >>> table.num_rows
    4
    >>> vecs.shape
    (4, 4)
    """
    vecs = unit_vectors(rows, dim, seed)
    ids = np.arange(rows, dtype=np.int64)
    labels = (ids % modulus).astype(np.int32)
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), dim)
    table = pa.table({"id": ids, "label": labels, "vector": vectors})
    return table, vecs


_TEXT_POOL_SIZE = 65_536
_EPOCH_MS = 1_700_000_000_000


def text_pool(seed: int) -> pa.Array:
    """
    Random lowercase strings to draw a text column from.

    Parameters
    ----------
    seed : int
        Generator seed.

    Returns
    -------
    pyarrow.Array
        65,536 strings, each 8 to 63 characters long.

    Examples
    --------
    >>> pool = text_pool(0)
    >>> len(pool)
    65536
    >>> all(8 <= len(s) < 64 for s in pool.to_pylist()[:100])
    True
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(8, 64, size=_TEXT_POOL_SIZE)
    letters = rng.integers(97, 123, size=int(lengths.sum()), dtype=np.uint8)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    return pa.StringArray.from_buffers(
        _TEXT_POOL_SIZE, pa.py_buffer(offsets), pa.py_buffer(letters.tobytes())
    )


def mixed_batch(
    start: int, rows: int, dim: int, seed: int, pool: pa.Array
) -> pa.RecordBatch:
    """
    One batch of a table with scalar, string, timestamp, and vector columns.

    Row ``i`` of the whole table depends only on ``i`` and ``seed``, so
    batches can be generated in any order.

    Parameters
    ----------
    start : int
        Id of the first row.
    rows : int
        Rows in the batch.
    dim : int
        Width of the ``vector`` column.
    seed : int
        Generator seed.
    pool : pyarrow.Array
        Strings for the ``text`` column, from :func:`text_pool`.

    Returns
    -------
    pyarrow.RecordBatch
        Columns ``id`` (int64), ``label`` (int32, every 10th null),
        ``ts`` (timestamp ms), ``price`` (float64), ``text`` (string),
        and ``vector`` (fixed-size list of float32).

    Examples
    --------
    >>> batch = mixed_batch(0, 4, 2, 0, text_pool(0))
    >>> batch.schema.names
    ['id', 'label', 'ts', 'price', 'text', 'vector']
    >>> batch.column("id").to_pylist()
    [0, 1, 2, 3]
    """
    rng = np.random.default_rng([seed, start])
    ids = np.arange(start, start + rows, dtype=np.int64)
    labels = pa.array(
        (ids % 1000).astype(np.int32), mask=(ids % 10 == 9), type=pa.int32()
    )
    ts = pa.array(_EPOCH_MS + ids * 1000, type=pa.timestamp("ms"))
    price = np.round(rng.lognormal(3.0, 1.0, size=rows), 2)
    text = pool.take(pa.array(rng.integers(0, len(pool), size=rows)))
    vecs = rng.standard_normal(rows * dim, dtype=np.float32)
    vector = pa.FixedSizeListArray.from_arrays(pa.array(vecs), dim)
    return pa.record_batch(
        {
            "id": ids,
            "label": labels,
            "ts": ts,
            "price": price,
            "text": text,
            "vector": vector,
        }
    )
