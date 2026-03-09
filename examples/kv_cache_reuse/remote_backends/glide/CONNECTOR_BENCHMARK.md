# LMCache Connector Benchmark: RESP vs GLIDE

## Motivation

LMCache stores KV cache tensors in a remote Valkey/Redis backend for cross-instance sharing. Each cache chunk is 32–80 MB depending on the model, and a single request may store or retrieve 32–256 chunks. At this scale, the connector's data path — how bytes move between Python memory and the network socket — dominates end-to-end latency.

LMCache currently supports two connector implementations for Valkey/Redis:

1. **RESPConnector** — a C++ RESP client (`lmcache_redis`) that performs all I/O in native threads with zero Python-side copies.
2. **ValkeyConnector** — a Python connector using the [Valkey GLIDE](https://github.com/valkey-io/valkey-glide) async client, which routes data through protobuf serialization and a Unix Domain Socket to a Rust core.

The GLIDE client offers a richer feature set (cluster support, connection management, observability) but its async architecture introduces overhead that limits throughput for large-value workloads. This benchmark quantifies that gap and evaluates whether targeted optimizations to the GLIDE sync client can close it.

## Connectors Under Test

### 1. RESPConnector (C++ baseline)

The RESPConnector wraps a C++ `LMCacheRedisClient` that manages N worker threads, each with its own socket connection. The Python layer submits batch operations (`batch_set`, `batch_get`) and is notified of completion via `eventfd` — the data never passes through Python heap memory.

- **SET path**: Python `memoryview` → C++ extracts raw pointer → `write()` to socket
- **GET path**: `read()` from socket → C++ writes directly into Python `memoryview`
- **Copies**: Zero between Python and the socket

### 2. ValkeyConnector (async GLIDE, as-is)

The current production ValkeyConnector uses GLIDE's async Python client (`GlideClient`). Each operation is serialized into a protobuf message, sent over a Unix Domain Socket to the Rust core, which then communicates with Valkey.

- **SET path**: `memoryview` → `bytes()` copy → protobuf serialization → UDS → Rust → socket
- **GET path**: socket → Rust → protobuf → UDS → Python `bytes` allocation → copy into target buffer
- **Copies**: 2–3 per operation for large values
- **Bottleneck**: Single UDS connection, protobuf overhead, Python `bytes` allocation under GIL

Additionally, the ValkeyConnector uses 2 keys per chunk (`:metadata` + `:kv_bytes`), doubling round-trips compared to RESP's single-key approach.

### 3. ValkeyConnector sync (optimized GLIDE, multiprocess)

This mode uses GLIDE's sync Python client (`glide_sync.GlideClient`) with two upstream optimizations submitted as PRs to `valkey-io/valkey-glide`:

- **[PR #5492](https://github.com/valkey-io/valkey-glide/pull/5492) — Zero-copy SET**: The sync client's CFFI layer (`ffi.from_buffer()`) already supports any buffer-protocol object, but the `isinstance` check only accepted `bytes`. Changing it to accept `memoryview` and `bytearray` lets tensor memory flow directly to the socket without an intermediate `bytes()` copy. Python-only change, two lines.

- **[PR #5493](https://github.com/valkey-io/valkey-glide/pull/5493) — Buffer GET**: A new `buffer` parameter on `get()` that writes the response directly into a caller-provided `memoryview`. Implemented via a new `command_with_buffer()` FFI function in Rust that uses `ptr::copy_nonoverlapping` from the response `Vec<u8>` into the target buffer — one copy instead of two, and no Python `bytes` allocation.

#### Multiprocess Architecture

The GLIDE sync client's CFFI layer holds the GIL during response handling, which causes deadlocks when multiple threads operate concurrently with large values (80+ MB). To achieve true parallelism, the benchmark uses N worker **processes** instead of threads:

- Each worker process creates its own `GlideClient` with its own Tokio runtime — completely independent, no GIL sharing
- KV data is passed via `multiprocessing.SharedMemory` — zero-copy across processes (same physical memory pages)
- Worker processes are created once at init time (persistent pool), not per-operation
- Control messages (key name, offset, length) are sent via the pool's internal pipe — only ~100 bytes per operation

This mirrors the RESP architecture: N independent connections doing I/O in parallel with no Python-level contention.

- **SET path**: `SharedMemory` → worker process → `ffi.from_buffer()` extracts pointer → Rust → socket (zero-copy)
- **GET path**: socket → Rust → `ptr::copy_nonoverlapping` into `SharedMemory` target (one copy)
- **Copies**: 0 for SET, 1 for GET

## Setup

| Component | Spec |
|-----------|------|
| EC2 client | `r7i.16xlarge` (64 vCPU, 512 GB RAM), `us-west-2b` |
| ElastiCache | `cache.r7g.2xlarge`, Valkey 8.2.0, same AZ, no TLS |
| Workers | 8 processes / connections |
| Python | 3.11 |

## Results

### Llama-3.1-8B, 8K context (32 MB × 32 keys = 1.1 GB)

| Connector | SET (GB/s) | GET (GB/s) | SET vs baseline | GET vs baseline |
|-----------|-----------|-----------|-----------------|-----------------|
| RESPConnector (C++ baseline) | 1.509 | 1.761 | — | — |
| ValkeyConnector sync (multiprocess) | 1.522 | 1.622 | **101%** | 92% |
| ValkeyConnector (async GLIDE) | 0.361 | 0.582 | 24% | 33% |

### Llama-3.1-70B, 8K context (84 MB × 32 keys = 2.7 GB)

| Connector | SET (GB/s) | GET (GB/s) | SET vs baseline | GET vs baseline |
|-----------|-----------|-----------|-----------------|-----------------|
| RESPConnector (C++ baseline) | 1.615 | 1.705 | — | — |
| ValkeyConnector sync (multiprocess) | 1.483 | 1.576 | 92% | 92% |
| ValkeyConnector (async GLIDE) | 0.325 | 0.515 | 20% | 30% |

### Llama-3.1-70B, 64K context (84 MB × 256 keys = 21.5 GB)

| Connector | SET (GB/s) | GET (GB/s) | SET vs baseline | GET vs baseline |
|-----------|-----------|-----------|-----------------|-----------------|
| RESPConnector (C++ baseline) | 1.536 | 1.741 | — | — |
| ValkeyConnector sync (multiprocess) | 1.607 | 1.740 | **105%** | **100%** |
| ValkeyConnector (async GLIDE) | 0.322 | 0.509 | 21% | 29% |

All data verified correct across all three modes and all workloads.

### Throughput Comparison

```
Llama-3.1-8B 8K — SET (GB/s)
─────────────────────────────────────────────────────
RESP (C++ baseline)  █████████████████████████████████████████████  1.509
GLIDE sync (opt.)    ██████████████████████████████████████████████  1.522
GLIDE async (as-is)  ███████████                                    0.361

Llama-3.1-70B 64K — SET (GB/s)
─────────────────────────────────────────────────────
RESP (C++ baseline)  █████████████████████████████████████████████  1.536
GLIDE sync (opt.)    ███████████████████████████████████████████████  1.607
GLIDE async (as-is)  ██████████                                     0.322

Llama-3.1-70B 64K — GET (GB/s)
─────────────────────────────────────────────────────
RESP (C++ baseline)  ██████████████████████████████████████████████  1.741
GLIDE sync (opt.)    ██████████████████████████████████████████████  1.740
GLIDE async (as-is)  █████████████                                  0.509
```

### Analysis

The async GLIDE client is **3–5× slower** than the C++ baseline across all workloads. This is caused by the single-UDS bottleneck, protobuf serialization, and mandatory `bytes` copies under the GIL.

The sync GLIDE client with multiprocessing and the two PRs reaches **92–105% of the C++ baseline**. Key observations:

- **GLIDE sync matches or exceeds RESP on SET**: For the 8B-8k and 70B-64k workloads, GLIDE sync SET is 101–105% of the C++ baseline. The zero-copy `ffi.from_buffer()` path is as efficient as the C++ pointer extraction, and the Rust core's connection handling may have slight advantages at scale.
- **GET is within 92–100%**: The remaining gap on GET is the single `ptr::copy_nonoverlapping` from Rust's response buffer into shared memory, vs RESP's direct socket-to-buffer read in C++.
- **Larger workloads perform better**: The 70B-64k workload (256 keys, 21.5 GB) shows the best relative performance because per-operation overhead (process dispatch, CFFI call setup) is amortized over more data.
- **Threading doesn't work** for large values: the GLIDE sync client's CFFI layer holds the GIL during response handling, causing deadlocks when multiple threads share the Tokio runtime with 80+ MB payloads. Multiprocessing with persistent worker pools avoids this entirely.

## Reproducing

```bash
python3.11 benchmark_connectors.py \
    --host <valkey-endpoint> \
    --port 6379 \
    --preset llama70b-64k \
    --num-workers 8
```

Available presets: `llama8b-8k`, `llama8b-64k`, `llama70b-8k`, `llama70b-64k`.

### Prerequisites

- LMCache installed from source (for C++ RESP extension)
- `valkey-glide` (async client, pip installable)
- Patched `glide_sync` with PRs [#5492](https://github.com/valkey-io/valkey-glide/pull/5492) and [#5493](https://github.com/valkey-io/valkey-glide/pull/5493)
- Valkey or Redis server accessible from the benchmark host
- For 64K context workloads: instance with ≥128 GB RAM (e.g., `r7i.16xlarge`)

## Implications for LMCache

These results demonstrate that the GLIDE sync client with two targeted optimizations can **match or exceed** the C++ RESP client's throughput, while gaining GLIDE's cluster support, connection management, and broader ecosystem integration. The required changes are:

1. **Upstream**: Merge PRs #5492 and #5493 into `valkey-glide` (both approved, awaiting second review).
2. **LMCache**: Implement a `SyncValkeyConnector` that uses persistent worker processes with `SharedMemory`:
   - N worker processes created at init, each with its own `glide_sync.GlideClient`
   - KV data passed via shared memory (zero-copy for SET, one copy for GET)
   - Control messages via lightweight IPC (queue/pipe) — only key names and offsets
   - This architecture matches the RESP client's design and achieves 92–105% of its throughput
