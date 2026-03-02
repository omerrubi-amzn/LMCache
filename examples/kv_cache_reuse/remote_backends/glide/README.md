# Glide Client Benchmark for LMCache ValkeyConnector

Benchmarks three Glide client configurations to find the best setup for the ValkeyConnector.

| # | Mode | Description |
|---|------|-------------|
| 1 | Async individual | Current ValkeyConnector behavior — one `set`/`get` per key, awaited sequentially |
| 2 | Async pipeline | `Batch(False)` for SET + `mget` for GET — batched round trips |
| 3 | Sync + ThreadPool | `glide_sync` client called from N threads |

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

# Custom parameters
python benchmark_glide_client.py \
    --host localhost \
    --port 6379 \
    --chunk-mb 4.0 \
    --num-workers 8 \
    --num-keys 500
```

Clear state between runs:

```bash
redis-cli -p 6379 FLUSHALL
```

## Comparing with RESP client

Run the RESP benchmark from `../resp/benchmark_resp_client.py` with the same parameters to compare Glide throughput against the C++ RESP client.
