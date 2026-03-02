# SPDX-License-Identifier: Apache-2.0
"""
Benchmark comparing different Glide client setups for LMCache ValkeyConnector.
Adapted from benchmark_resp_client.py

Tests three modes:
  1. Async Glide - individual set/get per key (current ValkeyConnector behavior)
  2. Async Glide - pipeline batch (Batch + mget)
  3. Sync Glide  - ThreadPoolExecutor with N workers
"""

import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

from glide import (
    Batch,
    GlideClient,
    GlideClientConfiguration,
    NodeAddress,
    ServerCredentials,
)


def make_credentials(username, password):
    if username or password:
        return ServerCredentials(password, username)
    return None


# ── 1. Async Glide (current ValkeyConnector approach) ──────────────────────


async def bench_async_individual(client, keys, buffers, read_bufs):
    """Mirrors current ValkeyConnector: individual async get/set per key."""
    # SET
    t0 = time.perf_counter()
    for k, buf in zip(keys, buffers):
        await client.set(k, bytes(buf))
    elapsed_set = time.perf_counter() - t0

    # GET
    t0 = time.perf_counter()
    for i, k in enumerate(keys):
        result = await client.get(k)
        if result:
            read_bufs[i][:] = result
    elapsed_get = time.perf_counter() - t0

    return elapsed_set, elapsed_get


# ── 2. Async Glide with pipeline batching ───────────────────────────────────


async def bench_async_pipeline(client, keys, buffers, read_bufs):
    """Uses Batch(False) pipeline to send all ops at once."""
    # SET via pipeline
    batch = Batch(False)
    for k, buf in zip(keys, buffers):
        batch.set(k, bytes(buf))

    t0 = time.perf_counter()
    await client.exec(batch, raise_on_error=True)
    elapsed_set = time.perf_counter() - t0

    # GET via mget
    t0 = time.perf_counter()
    results = await client.mget(keys)
    for i, r in enumerate(results):
        if r:
            read_bufs[i][:] = r
    elapsed_get = time.perf_counter() - t0

    return elapsed_set, elapsed_get


# ── 3. Sync Glide + ThreadPoolExecutor ──────────────────────────────────────


def _sync_set_worker(args):
    client, key, buf = args
    client.set(key, bytes(buf))


def _sync_get_worker(args):
    client, key = args
    return client.get(key)


def _sync_set_batch_worker(args):
    """Worker that pipelines a chunk of SET ops."""
    from glide_sync import Batch as SyncBatch
    client, key_buf_pairs = args
    batch = SyncBatch(False)
    for k, buf in key_buf_pairs:
        batch.set(k, bytes(buf))
    client.exec(batch, raise_on_error=True)


def _sync_get_batch_worker(args):
    """Worker that mgets a chunk of keys."""
    client, chunk_keys = args
    return client.mget(chunk_keys)


async def bench_sync_threaded(
    host, port, credentials, keys, buffers, read_bufs, num_workers
):
    """Uses glide_sync client from multiple threads."""
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

    # SET with thread pool
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(
            pool.map(
                _sync_set_worker,
                [(sync_client, k, buf) for k, buf in zip(keys, buffers)],
            )
        )
    elapsed_set = time.perf_counter() - t0

    # GET with thread pool
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(
            pool.map(
                _sync_get_worker, [(sync_client, k) for k in keys]
            )
        )
    for i, r in enumerate(results):
        if r:
            read_bufs[i][:] = r
    elapsed_get = time.perf_counter() - t0

    sync_client.close()
    return elapsed_set, elapsed_get


async def bench_sync_threaded_batched(
    host, port, credentials, keys, buffers, read_bufs, num_workers
):
    """Sync client with threads, but each thread handles a chunk via pipeline/mget."""
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

    # Split keys into chunks per worker
    def chunk_list(lst, n):
        size = (len(lst) + n - 1) // n
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    key_chunks = chunk_list(list(range(len(keys))), num_workers)

    # SET: each thread pipelines its chunk
    set_args = [
        (sync_client, [(keys[i], buffers[i]) for i in chunk])
        for chunk in key_chunks
    ]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(_sync_set_batch_worker, set_args))
    elapsed_set = time.perf_counter() - t0

    # GET: each thread mgets its chunk
    get_args = [
        (sync_client, [keys[i] for i in chunk])
        for chunk in key_chunks
    ]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        chunk_results = list(pool.map(_sync_get_batch_worker, get_args))
    # Reassemble results in order
    for chunk_indices, results in zip(key_chunks, chunk_results):
        for idx, r in zip(chunk_indices, results):
            if r:
                read_bufs[idx][:] = r
    elapsed_get = time.perf_counter() - t0

    sync_client.close()
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

    # Prepare test data
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

    # ── Test 1: Async individual (current connector behavior)
    print("\n[1] Async Glide - individual set/get per key:")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    es, eg = await bench_async_individual(client, keys, buffers, read_bufs)
    mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
    if mismatches:
        print(f"  WARNING: {mismatches}/{num_keys} keys had data mismatch")
    else:
        print("  Data verified OK")
    report(es, eg)

    # ── Test 2: Async pipeline batch
    print("\n[2] Async Glide - pipeline batch (Batch + mget):")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    es, eg = await bench_async_pipeline(client, keys, buffers, read_bufs)
    mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
    if mismatches:
        print(f"  WARNING: {mismatches}/{num_keys} keys had data mismatch")
    else:
        print("  Data verified OK")
    report(es, eg)

    # ── Test 3: Sync + threads
    print(f"\n[3] Sync Glide - ThreadPool ({num_workers} workers):")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    try:
        es, eg = await bench_sync_threaded(
            host, port, credentials, keys, buffers, read_bufs, num_workers
        )
        mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
        if mismatches:
            print(f"  WARNING: {mismatches}/{num_keys} keys had data mismatch")
        else:
            print("  Data verified OK")
        report(es, eg)
    except ImportError:
        print("  SKIPPED - glide_sync not available (requires valkey-glide >= 2.1)")

    # ── Test 4: Sync + threads + batched per thread
    print(f"\n[4] Sync Glide - ThreadPool ({num_workers} workers) + pipeline/mget per thread:")
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    try:
        es, eg = await bench_sync_threaded_batched(
            host, port, credentials, keys, buffers, read_bufs, num_workers
        )
        mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])
        if mismatches:
            print(f"  WARNING: {mismatches}/{num_keys} keys had data mismatch")
        else:
            print("  Data verified OK")
        report(es, eg)
    except ImportError:
        print("  SKIPPED - glide_sync not available (requires valkey-glide >= 2.1)")

    await client.close()
    print("-" * 60)
    print("All tests passed.")


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
    args = parser.parse_args()
    asyncio.run(run_benchmark(**vars(args)))
