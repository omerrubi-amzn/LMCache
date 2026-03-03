# Glide Client Benchmark for LMCache ValkeyConnector

Benchmarks old vs new ValkeyConnector patterns to measure the impact of
single-key packing and batched operations.

| # | Mode | Description |
|---|------|-------------|
| 1 | Old async individual | 2 keys/chunk (metadata + kv_bytes), sequential — original ValkeyConnector |
| 2 | Old async pipeline | 2 keys/chunk, batched pipeline |
| 3 | New async individual | 1 key/chunk (packed metadata header), sequential |
| 4 | New async pipeline | 1 key/chunk, `Batch(False)` for SET + `mget` for GET |
| 5 | New sync + ThreadPool | 1 key/chunk, `glide_sync` client with N threads |
| 6 | New sync + N clients | 1 key/chunk, N independent `GlideClient` instances tiled across threads (RESP-like) |

## Prerequisites

```bash
pip install valkey-glide   # >= 2.1 for sync client support
```

## Quickstart

Start Valkey/Redis with multiple IO threads:

```bash
# Using Valkey
valkey-server --protected-mode no --save '' --appendonly no --io-threads 4

# Or using Redis
redis-server --protected-mode no --save '' --appendonly no --io-threads 4
```

Run the benchmark:

```bash
# Defaults: host=127.0.0.1, port=6379, chunk-mb=4.0, num-workers=8, num-keys=500
python benchmark_glide_client.py

# Use a model preset for realistic chunk sizes
python benchmark_glide_client.py --preset llama8b-8k    # 32MB chunks, 32 keys
python benchmark_glide_client.py --preset llama8b-64k   # 32MB chunks, 256 keys
python benchmark_glide_client.py --preset llama70b-8k   # 80MB chunks, 32 keys
python benchmark_glide_client.py --preset llama70b-64k  # 80MB chunks, 256 keys

# Custom parameters (override preset values)
python benchmark_glide_client.py --preset llama8b-8k --num-workers 16
```

### Presets

Presets set `--chunk-mb` and `--num-keys` to match real LMCache workloads
(256 tokens/chunk, GQA with 8 KV heads, 128 head dim, fp16):

| Preset | Chunk Size | Chunks | Total Data | Simulates |
|--------|-----------|--------|------------|-----------|
| `llama8b-8k` | 32 MB | 32 | 1 GB | 8K context on Llama-3.1-8B |
| `llama8b-64k` | 32 MB | 256 | 8 GB | 64K context on Llama-3.1-8B |
| `llama70b-8k` | 80 MB | 32 | 2.5 GB | 8K context on Llama-3.1-70B |
| `llama70b-64k` | 80 MB | 256 | 20 GB | 64K context on Llama-3.1-70B |

Clear state between runs:

```bash
redis-cli -p 6379 FLUSHALL
```

## Comparing with RESP client

Run the RESP benchmark from `benchmark_resp_standalone.py` with the same
parameters to compare Glide throughput against the C++ RESP client.
