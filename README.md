# bench_lance

Benchmarks for the Lance file format and embedded LanceDB. Pixi installs the environment.

```bash
pixi install
pixi run test
pixi run format
pixi run vector
pixi run convert
pixi run bench
```

Extra arguments pass through:

```bash
pixi run vector -- --rows 50000 --queries 200 --index ivf_pq --out results/vector.json
pixi run format -- --rows 1000000 --take 1000 --workdir /tmp/lance-format
pixi run convert -- --rows 67108864 --threads 1,8,64 --workdir /data/convert
```

`format` writes one synthetic table as Lance and as Parquet, then times a warm-cache full scan, a filtered column read (`id % 100 == 0`), and a random take. The Parquet take decodes the file. The Lance take reads the requested rows.

`convert` writes a mixed table (`id`, nullable `label`, `ts`, `price`, `text`, and a `vector` of `--dim` floats) as one Parquet file with `--group-rows` rows per row group, then converts it to Lance and back at each `--threads` count. Each run is a child process with `LANCE_CPU_THREADS` and Arrow's CPU pool set to the thread count. `*_single` uses one writer: `lance.write_dataset` for Parquet to Lance, one `ParquetWriter` for Lance to Parquet. `*_parallel` gives each thread a contiguous slice of row groups or fragments. Parquet to Lance commits the fragments as one version, and Lance to Parquet writes one file per thread. `parquet_scan` and `lance_scan` read without writing. Sources are warm in the page cache. `wall_s` ends when the writer returns. `sync_s` is the `os.sync` after it, so on a slow disk most of the cost is there. Sources are reused while `--rows`, `--dim`, `--seed`, and `--group-rows` match.

`vector` loads unit vectors into LanceDB, times ingest and index build, then sweeps `nprobes` and `refine_factor`. Recall is the mean overlap with an exact search (`bypass_vector_index`) at the same `k`. A filter point (`label = 0`) uses the same ground-truth rule. Latency is one serial client that returns ids only. Query vectors are not inserted.

The default index is `IVF_HNSW_SQ` (cosine, `m=16`, `ef_construction=150`). `--index ivf_pq` builds an `IVF_PQ` index. `--partitions 0` keeps at least 256 rows per partition so k-means training fits in the table. Reported QPS is serial, from one Python process.

Defaults are 100,000 rows for `format` and 20,000 rows for `vector`. `all` uses 20,000 rows for both. Results are printed as text. `--out` also writes JSON.
