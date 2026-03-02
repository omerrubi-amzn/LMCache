# LMCache Remote Backend Client Benchmark

## Overview

We benchmarked the throughput of different client configurations for LMCache's remote KV cache storage, comparing the existing C++ RESP client against multiple Valkey GLIDE client setups. The goal is to evaluate whether GLIDE can serve as a viable connector and identify the optimal configuration.

## Test Setup

- Redis 7.0.7 with 4 IO threads, localhost, no persistence (`--save '' --appendonly no`)
- Python 3.11, `valkey-glide` 2.2.7, `valkey-glide-sync` 2.2.7
- Workload: 500 keys × 4MB values (2GB total), matching LMCache's recommended chunk size for high-throughput KV cache transfers
- Each test performs a full SET of all keys followed by a full GET, with data integrity verification
- FLUSHALL (blocking) issued between each test to ensure clean state

## Results

| Client Configuration | SET (GB/s) | GET (GB/s) |
|---------------------|------------|------------|
| RESP (C++) | 1.63 | 0.81 |
| GLIDE sync + ThreadPool | 1.48 | 0.87 |
| GLIDE async pipeline | 0.33 | 0.41 |
| GLIDE async individual | 0.29 | 0.58 |

## Client Configurations

### RESP C++ Client

LMCache's custom C++ implementation using raw RESP protocol with multithreaded socket I/O. Each worker thread opens its own connection and performs direct socket reads/writes. Keys are tiled across workers. This is the current high-performance path and serves as the baseline.

### GLIDE Async Individual

Mirrors the current ValkeyConnector implementation from [PR #1743](https://github.com/LMCache/LMCache/pull/1743). Uses the async `glide` module with individual `await client.set()`/`await client.get()` calls per key, sequentially. Single-threaded, single connection multiplexed through GLIDE's Rust core.

### GLIDE Async Pipeline

Uses `Batch(False)` (non-atomic pipeline) to batch all SET operations into a single round trip, and `mget` for bulk GET. Still single-threaded async, but reduces round-trip overhead.

### GLIDE Sync + ThreadPool

Uses the `glide_sync` module with a `ThreadPoolExecutor` (8 workers). Each thread issues individual `set()`/`get()` calls concurrently. GLIDE's Rust core handles connection multiplexing internally.

## Analysis

- **SET throughput** is where configurations diverge most. The sync threaded approach achieves 5x the throughput of the current async implementation, getting within ~10% of the C++ baseline.
- **GET throughput** is ~0.8 GB/s for the best configurations. The sync threaded GLIDE client slightly outperforms the C++ RESP client (0.87 vs 0.81 GB/s).
- **Pipelining/mget did not help** at this value size. With 4MB values, individual operations across threads allow Redis to interleave responses across its IO threads more effectively than a single large batched response.
- The C++ RESP client's SET advantage comes from bypassing the Python GIL entirely — each C++ worker thread does raw socket I/O independently.

## Benchmark Scripts

Located in `examples/kv_cache_reuse/remote_backends/glide/`:
- `benchmark_glide_client.py` — runs all four GLIDE configurations
- `benchmark_resp_standalone.py` — runs the C++ RESP client (requires compiling `csrc/redis/` with pybind11)

Both scripts accept `--host`, `--port`, `--chunk-mb`, `--num-workers`, `--num-keys` for consistent parameterization.
