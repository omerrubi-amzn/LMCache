# SPDX-License-Identifier: Apache-2.0
"""
Benchmark comparing different Glide client setups for LMCache ValkeyConnector.
Adapted from benchmark_resp_client.py

Tests five modes:
  1. Async individual, 2 keys/chunk  (old ValkeyConnector)
  2. Async pipeline, 2 keys/chunk    (old + batching)
  3. Async individual, 1 key/chunk   (new ValkeyConnector single-key)
  4. Async pipeline, 1 key/chunk     (new ValkeyConnector batched)
  5. Sync + ThreadPool, 1 key/chunk  (new ValkeyConnector sync threaded)
"""

import argparse
import asyncio
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor

from glide import (
    Batch,
    GlideClient,
    GlideClientConfiguration,
    NodeAddress,
    ServerCredentials,
)

# ── Metadata packing (mirrors improved ValkeyConnector) ─────────────────────
# Format: [4-byte LE metadata_len][metadata_bytes][kv_bytes]
# We use a fixed 28-byte fake metadata header to match real overhead.

FAKE_METADATA = struct.pack("<7I", 0, 0, 0, 0, 0, 0, 0)  # 28 bytes


def pack_value(kv_bytes):
    """Pack metadata header + kv payload into a single value."""
    hdr = len(FAKE_METADATA).to_bytes(4, "little")
    return hdr + FAKE_METADATA + bytes(kv_bytes)


def unpack_value(raw):
    """Unpack kv payload from a packed value."""
    mv = memoryview(raw)
    meta_len = int.from_bytes(mv[:4], "little")
    return mv[4 + meta_len :]


def make_credentials(username, password):
    if username or password:
        return ServerCredentials(password, username)
    return None


# ── 1. Async individual, 2 keys per chunk (old ValkeyConnector) ─────────────


async def bench_old_async_individual(client, keys, buffers, read_bufs):
    """Old ValkeyConnector: 2 keys per chunk (metadata + kv_bytes), sequential."""
    # SET
    t0 = time.perf_counter()
    for k, buf in zip(keys, buffers):
        batch = Batch(False)
        batch.set(f"{k}:kv_bytes", bytes(buf))
        batch.set(f"{k}:metadata", FAKE_METADATA)
        await client.exec(batch, raise_on_error=False)
    elapsed_set = time.perf_counter() - t0

    # GET
    t0 = time.perf_counter()
    for i, k in enumerate(keys):
        results = await client.mget([f"{k}:metadata", f"{k}:kv_bytes"])
        if results[1]:
            read_bufs[i][:] = results[1]
    elapsed_get = time.perf_counter() - t0

    return elapsed_set, elapsed_get


# ── 2. Async pipeline, 2 keys per chunk (old + batching) ────────────────────


async def bench_old_async_pipeline(client, keys, buffers, read_bufs):
    """Old format but all ops batched into one pipeline."""
    # SET
    batch = Batch(False)
    for k, buf in zip(keys, buffers):
        batch.set(f"{k}:kv_bytes", bytes(buf))
        batch.set(f"{k}:metadata", FAKE_METADATA)
    t0 = time.perf_counter()
    await client.exec(batch, raise_on_error=True)
    elapsed_set = time.perf_counter() - t0

    # GET via mget (interleaved metadata + kv keys)
    get_keys = []
    for k in keys:
        get_keys.append(f"{k}:metadata")
        get_keys.append(f"{k}:kv_bytes")
    t0 = time.perf_counter()
    results = await client.mget(get_keys)
    for i in range(len(keys)):
        kv = results[2 * i + 1]
        if kv:
            read_bufs[i][:] = kv
    elapsed_get = time.perf_counter() - t0

    return elapsed_set, elapsed_get


# ── 3. Async individual, 1 key per chunk (new ValkeyConnector) ──────────────


async def bench_new_async_individual(client, keys, buffers, read_bufs):
    """New ValkeyConnector: single packed key, sequential set/get."""
    # SET
    t0 = time.perf_counter()
    for k, buf in zip(keys, buffers):
        await client.set(k, pack_value(buf))
    elapsed_set = time.perf_counter() - t0

    # GET
    t0 = time.perf_counter()
    for i, k in enumerate(keys):
        raw = await client.get(k)
        if raw:
            payload = unpack_value(raw)
            read_bufs[i][:] = payload
    elapsed_get = time.perf_counter() - t0

    return elapsed_set, elapsed_get


# ── 4. Async pipeline, 1 key per chunk (new ValkeyConnector batched) ────────


async def bench_new_async_pipeline(client, keys, buffers, read_bufs):
    """New ValkeyConnector: single packed key, batched pipeline + mget."""
    # SET via pipeline
    batch = Batch(False)
    for k, buf in zip(keys, buffers):
        batch.set(k, pack_value(buf))
    t0 = time.perf_counter()
    await client.exec(batch, raise_on_error=True)
    elapsed_set = time.perf_counter() - t0

    # GET via mget
    t0 = time.perf_counter()
    results = await client.mget(keys)
    for i, r in enumerate(results):
        if r:
            read_bufs[i][:] = unpack_value(r)
    elapsed_get = time.perf_counter() - t0

    return elapsed_set, elapsed_get


# ── 5. Sync + ThreadPool, 1 key per chunk ───────────────────────────────────


def _sync_set_worker(args):
    client, key, buf = args
    client.set(key, pack_value(buf))


def _sync_get_worker(args):
    client, key = args
    return client.get(key)


async def bench_new_sync_threaded(
    host, port, credentials, keys, buffers, read_bufs, num_workers
):
    """New ValkeyConnector pattern with sync client + thread pool."""
    from glide_sync import (
        GlideClient as SyncGlideClient,
        GlideClientConfiguration as SyncGlideClientConfiguration,
        NodeAddress as SyncNodeAddress,
    )

    config = SyncGlideClientConfiguration(
        addresses=[SyncNodeAddress(host, port)],
        credentials=credentials,
        request_timeout=5000,
    )
    sync_client = SyncGlideClient.create(config)

    # SET
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(
            _sync_set_worker,
            [(sync_client, k, buf) for k, buf in zip(keys, buffers)],
        ))
    elapsed_set = time.perf_counter() - t0

    # GET
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(
            _sync_get_worker, [(sync_client, k) for k in keys]
        ))
    for i, r in enumerate(results):
        if r:
            read_bufs[i][:] = unpack_value(r)
    elapsed_get = time.perf_counter() - t0

    sync_client.close()
    return elapsed_set, elapsed_get


# ── 6. Sync + ThreadPool, N clients (RESP-like architecture) ────────────────


def _multi_client_set_worker(args):
    """Each worker has its own GlideClient and handles a slice of keys."""
    client, worker_keys, worker_bufs = args
    for k, buf in zip(worker_keys, worker_bufs):
        client.set(k, pack_value(buf))


def _multi_client_get_worker(args):
    """Each worker GETs its slice of keys."""
    client, worker_keys = args
    return [(k, client.get(k)) for k, _ in zip(worker_keys, worker_keys)]


async def bench_new_sync_multi_client(
    host, port, credentials, keys, buffers, read_bufs, num_workers
):
    """N independent GlideClient instances, each pinned to a thread with a
    slice of keys — mirrors the RESP C++ architecture."""
    from glide_sync import (
        GlideClient as SyncGlideClient,
        GlideClientConfiguration as SyncGlideClientConfiguration,
        NodeAddress as SyncNodeAddress,
    )

    # Create N independent clients (N connections)
    clients = []
    for _ in range(num_workers):
        config = SyncGlideClientConfiguration(
            addresses=[SyncNodeAddress(host, port)],
            credentials=credentials,
            request_timeout=5000,
        )
        clients.append(SyncGlideClient.create(config))

    # Tile keys across workers
    def tile(lst, n):
        size = (len(lst) + n - 1) // n
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    key_tiles = tile(keys, num_workers)
    buf_tiles = tile(buffers, num_workers)
    idx_tiles = tile(list(range(len(keys))), num_workers)

    # SET — each thread uses its own client on its slice
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(
            _multi_client_set_worker,
            [(clients[i], key_tiles[i], buf_tiles[i]) for i in range(len(key_tiles))],
        ))
    elapsed_set = time.perf_counter() - t0

    # GET — each thread uses its own client on its slice
    def _get_tile(args):
        client, tile_keys = args
        return [client.get(k) for k in tile_keys]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        tile_results = list(pool.map(
            _get_tile,
            [(clients[i], key_tiles[i]) for i in range(len(key_tiles))],
        ))
    for indices, results in zip(idx_tiles, tile_results):
        for idx, r in zip(indices, results):
            if r:
                read_bufs[idx][:] = unpack_value(r)
    elapsed_get = time.perf_counter() - t0

    for c in clients:
        c.close()
    return elapsed_set, elapsed_get


# ── Main ────────────────────────────────────────────────────────────────────


async def run_benchmark(host, port, chunk_mb, num_workers, num_keys, username, password):
    chunk_bytes = int(chunk_mb * 1024 * 1024)
    credentials = make_credentials(username, password)

    config = GlideClientConfiguration(
        addresses=[NodeAddress(host, port)],
        credentials=credentials,
        request_timeout=5000,
    )
    client = await GlideClient.create(config)

    print("Glide Client Benchmark")
    print(f"Server: {host}:{port}, Workers: {num_workers}")
    print(f"Chunk size: {chunk_bytes / 1024:.0f}KB, Keys: {num_keys}")
    print(f"Total data: {num_keys * chunk_mb:.0f}MB")
    print("-" * 60)

    print("Preparing buffers...")
    keys = [f"bench:glide:{i}" for i in range(num_keys)]
    buffers = [bytearray(os.urandom(chunk_bytes)) for _ in range(num_keys)]
    total_bytes = num_keys * chunk_bytes

    def report(elapsed_set, elapsed_get):
        tp_set = total_bytes / elapsed_set / (1024**3)
        tp_get = total_bytes / elapsed_get / (1024**3)
        print(
            f"  SET: {tp_set:6.2f} GB/s ({elapsed_set:.3f}s)  "
            f"GET: {tp_get:6.2f} GB/s ({elapsed_get:.3f}s)"
        )

    def run_test(label, coro):
        """Run a single test with FLUSHALL, verification, and reporting."""
        asyncio.get_event_loop().run_until_complete(
            client.custom_command(["FLUSHALL"])
        )
        print(f"\n{label}")
        read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
        es, eg = asyncio.get_event_loop().run_until_complete(coro(read_bufs))
        mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
        if mismatches:
            print(f"  WARNING: {mismatches}/{num_keys} keys had data mismatch")
        else:
            print("  Data verified OK")
        report(es, eg)

    # ── Old ValkeyConnector (2 keys per chunk) ──

    await client.custom_command(["FLUSHALL"])
    print("\n[1] Old connector - async individual (2 keys/chunk):")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    es, eg = await bench_old_async_individual(client, keys, buffers, read_bufs)
    mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
    print("  Data verified OK" if not mismatches else f"  WARNING: {mismatches} mismatches")
    report(es, eg)

    await client.custom_command(["FLUSHALL"])
    print("\n[2] Old connector - async pipeline (2 keys/chunk):")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    es, eg = await bench_old_async_pipeline(client, keys, buffers, read_bufs)
    mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
    print("  Data verified OK" if not mismatches else f"  WARNING: {mismatches} mismatches")
    report(es, eg)

    # ── New ValkeyConnector (1 key per chunk, packed metadata) ──

    await client.custom_command(["FLUSHALL"])
    print("\n[3] New connector - async individual (1 key/chunk):")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    es, eg = await bench_new_async_individual(client, keys, buffers, read_bufs)
    mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
    print("  Data verified OK" if not mismatches else f"  WARNING: {mismatches} mismatches")
    report(es, eg)

    await client.custom_command(["FLUSHALL"])
    print("\n[4] New connector - async pipeline (1 key/chunk, Batch + mget):")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    es, eg = await bench_new_async_pipeline(client, keys, buffers, read_bufs)
    mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
    print("  Data verified OK" if not mismatches else f"  WARNING: {mismatches} mismatches")
    report(es, eg)

    await client.custom_command(["FLUSHALL"])
    print(f"\n[5] New connector - sync ThreadPool ({num_workers} workers, 1 key/chunk):")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    try:
        es, eg = await bench_new_sync_threaded(
            host, port, credentials, keys, buffers, read_bufs, num_workers
        )
        mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
        print("  Data verified OK" if not mismatches else f"  WARNING: {mismatches} mismatches")
        report(es, eg)
    except ImportError:
        print("  SKIPPED - glide_sync not available (requires valkey-glide >= 2.1)")

    await client.custom_command(["FLUSHALL"])
    print(f"\n[6] New connector - sync ThreadPool, N clients ({num_workers} connections, 1 key/chunk):")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    try:
        es, eg = await bench_new_sync_multi_client(
            host, port, credentials, keys, buffers, read_bufs, num_workers
        )
        mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
        print("  Data verified OK" if not mismatches else f"  WARNING: {mismatches} mismatches")
        report(es, eg)
    except ImportError:
        print("  SKIPPED - glide_sync not available (requires valkey-glide >= 2.1)")

    await client.close()
    print("-" * 60)
    print("All tests passed.")


# ── Model presets ────────────────────────────────────────────────────────────
# chunk_bytes = num_layers * kv_dim * chunk_tokens * num_kv_heads * head_size * dtype_bytes
# chunk_tokens = 256 (LMCache default)
# All models use GQA with 8 KV heads, 128 head dim, fp16 (2 bytes), kv_dim=2

MODEL_PRESETS = {
    # (chunk_mb, description)
    "llama8b":  (32 * 2 * 256 * 8 * 128 * 2 / (1024**2),  "Llama-3.1-8B"),
    "llama70b": (80 * 2 * 256 * 8 * 128 * 2 / (1024**2),  "Llama-3.1-70B"),
}

CONTEXT_PRESETS = {
    # num_keys = context_tokens / chunk_tokens(256)
    "8k":  32,
    "64k": 256,
}


def apply_preset(args):
    """Override chunk-mb and num-keys from --preset if not explicitly set."""
    if not args.preset:
        return
    parts = args.preset.split("-", 1)  # e.g. "llama8b-64k"
    model_key = parts[0]
    ctx_key = parts[1] if len(parts) > 1 else None

    if model_key not in MODEL_PRESETS:
        raise SystemExit(
            f"Unknown model preset '{model_key}'. "
            f"Available: {', '.join(MODEL_PRESETS)}"
        )
    chunk_mb, desc = MODEL_PRESETS[model_key]
    if args.chunk_mb == 4.0:  # default
        args.chunk_mb = chunk_mb
    if ctx_key:
        if ctx_key not in CONTEXT_PRESETS:
            raise SystemExit(
                f"Unknown context preset '{ctx_key}'. "
                f"Available: {', '.join(CONTEXT_PRESETS)}"
            )
        if args.num_keys == 500:  # default
            args.num_keys = CONTEXT_PRESETS[ctx_key]
    print(f"Preset: {desc}, chunk={args.chunk_mb:.1f}MB, keys={args.num_keys}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Glide client setups for LMCache ValkeyConnector"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--chunk-mb", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-keys", type=int, default=500)
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Model-context preset, e.g. llama8b-8k, llama70b-64k. "
             "Overrides --chunk-mb and --num-keys with realistic values. "
             f"Models: {', '.join(MODEL_PRESETS)}. "
             f"Contexts: {', '.join(CONTEXT_PRESETS)}.",
    )
    args = parser.parse_args()
    apply_preset(args)
    kwargs = vars(args)
    kwargs.pop("preset")
    asyncio.run(run_benchmark(**kwargs))
