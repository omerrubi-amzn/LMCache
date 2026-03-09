# SPDX-License-Identifier: Apache-2.0
"""
Benchmark comparing LMCache connector implementations:

  1. RESPConnector  — C++ RESP client (baseline)
  2. ValkeyConnector — async GLIDE client (current production)
  3. ValkeyConnector (sync optimized) — GLIDE sync client with
     zero-copy SET (memoryview passthrough) and buffer GET
     (direct-to-buffer reads), using a ThreadPoolExecutor.

All three use the same LMCache abstractions (CacheEngineKey, MemoryObj,
LocalCPUBackend) so the comparison is apples-to-apples at the connector
level.

Usage:
    python benchmark_connectors.py \
        --host <valkey-host> --port 6379 \
        --num-keys 32 --num-workers 8 \
        [--preset llama8b-8k]

Prerequisites:
    - Valkey/Redis server running at --host:--port
    - lmcache installed from source (for C++ RESP extension)
    - valkey-glide with sync client patches (PRs #5492, #5493)
"""

# Standard
import argparse
import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryObj,
    PinMemoryAllocator,
    TensorMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import LocalCPUBackend

# ── Model presets ────────────────────────────────────────────────────────────
# chunk_bytes = num_layers * kv_dim * chunk_tokens * num_kv_heads
#               * head_dim * dtype_bytes
# kv_dim = 2 (key + value), chunk_tokens = 256, 8 KV heads, 128 head dim,
# fp16 = 2 bytes

MODEL_PRESETS = {
    "llama8b": {
        "num_layers": 32,
        "chunk_tokens": 256,
        "num_kv_heads": 8,
        "head_dim": 128,
        "desc": "Llama-3.1-8B",
    },
    "llama70b": {
        "num_layers": 80,
        "chunk_tokens": 256,
        "num_kv_heads": 8,
        "head_dim": 128,
        "desc": "Llama-3.1-70B",
    },
}

CONTEXT_PRESETS = {
    "8k": 32,    # 8192 / 256 = 32 chunks
    "64k": 256,  # 65536 / 256 = 256 chunks
}


def get_preset_params(preset_str):
    """Parse preset like 'llama8b-8k' into (model_params, num_keys)."""
    parts = preset_str.split("-", 1)
    model_key = parts[0]
    ctx_key = parts[1] if len(parts) > 1 else "8k"
    if model_key not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Available: {', '.join(MODEL_PRESETS)}"
        )
    if ctx_key not in CONTEXT_PRESETS:
        raise ValueError(
            f"Unknown context '{ctx_key}'. "
            f"Available: {', '.join(CONTEXT_PRESETS)}"
        )
    return MODEL_PRESETS[model_key], CONTEXT_PRESETS[ctx_key]


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_metadata(num_layers, chunk_tokens, num_kv_heads, head_dim):
    """Create LMCacheMetadata matching the model preset."""
    kv_shape = (num_layers, 2, chunk_tokens, num_kv_heads, head_dim)
    return LMCacheMetadata(
        model_name="bench-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        use_mla=False,
        chunk_size=chunk_tokens,
    )


def make_key(i: int) -> CacheEngineKey:
    """Create a unique CacheEngineKey for index i."""
    return CacheEngineKey(
        model_name="bench-model",
        world_size=1,
        worker_id=0,
        chunk_hash=i,
        dtype=torch.bfloat16,
    )


def fill_memory_obj(memory_obj: MemoryObj, seed: int):
    """Fill a MemoryObj with deterministic random data."""
    torch.manual_seed(seed)
    if hasattr(memory_obj, 'raw_data') and hasattr(memory_obj.raw_data, 'shape'):
        test_data = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_data.to(torch.float32).to(torch.bfloat16))


def init_asyncio_loop():
    """Start a background asyncio event loop."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def close_asyncio_loop(loop, thread):
    """Shut down the background event loop."""
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread.is_alive():
        thread.join(timeout=2.0)
    if not loop.is_closed():
        loop.close()


def flush_server(host, port):
    """FLUSHALL via a temporary GLIDE connection."""
    from glide import GlideClient, GlideClientConfiguration, NodeAddress

    async def _flush():
        config = GlideClientConfiguration(
            addresses=[NodeAddress(host, port)],
            request_timeout=30000,
        )
        client = await GlideClient.create(config)
        await client.custom_command(["FLUSHALL"])
        await client.close()

    asyncio.run(_flush())


# ── Benchmark: RESPConnector ────────────────────────────────────────────────


def bench_resp_connector(
    host, port, num_workers, keys, memory_objs, loop
):
    """Benchmark RESPConnector (C++ RESP client)."""
    from lmcache.v1.storage_backend.resp_client import RESPClient

    num_keys = len(keys)
    chunk_bytes = memory_objs[0].get_size()
    total_bytes = num_keys * chunk_bytes

    print("    Flushing server...", flush=True)
    flush_server(host, port)

    # Use RESPClient directly (like benchmark_resp_client.py does)
    client = RESPClient(host, port, num_workers, loop=loop)
    print("    RESP client created", flush=True)

    key_strs = [k.to_string() for k in keys]
    send_bufs = [
        memoryview(mobj.byte_array) if not isinstance(mobj.byte_array, memoryview)
        else mobj.byte_array.cast("B") if mobj.byte_array.format != "B"
        else mobj.byte_array
        for mobj in memory_objs
    ]

    # SET (batched)
    print("    RESP SET starting...", flush=True)
    t0 = time.perf_counter()
    fut = asyncio.run_coroutine_threadsafe(
        client.batch_set(key_strs, send_bufs), loop
    )
    fut.result(timeout=60)
    set_elapsed = time.perf_counter() - t0
    set_gbps = total_bytes / set_elapsed / 1e9
    print(f"    RESP SET done: {set_elapsed:.1f}s", flush=True)

    # GET (batched)
    print("    RESP GET starting...", flush=True)
    recv_bufs = [memoryview(bytearray(chunk_bytes)) for _ in range(num_keys)]
    t0 = time.perf_counter()
    fut = asyncio.run_coroutine_threadsafe(
        client.batch_get(key_strs, recv_bufs), loop
    )
    fut.result(timeout=60)
    get_elapsed = time.perf_counter() - t0
    get_gbps = total_bytes / get_elapsed / 1e9
    print(f"    RESP GET done: {get_elapsed:.1f}s", flush=True)

    # Verify
    mismatches = sum(
        1 for i in range(num_keys)
        if bytes(recv_bufs[i]) != bytes(send_bufs[i][:chunk_bytes])
    )

    client.close()

    return set_gbps, get_gbps, mismatches


# ── Benchmark: ValkeyConnector (async, as-is) ───────────────────────────────


def bench_valkey_connector(
    host, port, keys, memory_objs, loop
):
    """Benchmark async GLIDE client (current ValkeyConnector pattern)."""
    from glide import (
        Batch,
        GlideClient,
        GlideClientConfiguration,
        NodeAddress,
    )
    from lmcache.v1.protocol import RemoteMetadata

    num_keys = len(keys)
    chunk_bytes = memory_objs[0].get_size()
    total_bytes = num_keys * chunk_bytes

    async def _create():
        config = GlideClientConfiguration(
            addresses=[NodeAddress(host, port)],
            request_timeout=120000,
        )
        return await GlideClient.create(config)

    client = asyncio.run_coroutine_threadsafe(_create(), loop).result(timeout=5)
    print("    Valkey async client created", flush=True)

    def get_keys(key):
        key_str = key.to_string()
        return f"{key_str}:metadata", f"{key_str}:kv_bytes"

    # SET (same pattern as ValkeyConnector._put)
    print("    Flushing server...", flush=True)
    flush_server(host, port)
    print("    Valkey SET starting...", flush=True)

    async def do_set():
        for key, mobj in zip(keys, memory_objs):
            metadata_key, kv_key = get_keys(key)
            kv_bytes = mobj.byte_array
            if not isinstance(kv_bytes, memoryview):
                kv_bytes = memoryview(kv_bytes)
            elif kv_bytes.format != "B":
                kv_bytes = kv_bytes.cast("B")
            kv_shapes = mobj.get_shapes()
            kv_dtypes = mobj.get_dtypes()
            memory_format = mobj.get_memory_format()
            metadata_bytes = RemoteMetadata(
                len(kv_bytes), kv_shapes, kv_dtypes, memory_format
            ).serialize()
            batch = Batch(False)
            batch.set(kv_key, bytes(kv_bytes))
            batch.set(metadata_key, metadata_bytes)
            await client.exec(batch, raise_on_error=False)

    t0 = time.perf_counter()
    asyncio.run_coroutine_threadsafe(do_set(), loop).result(timeout=120)
    set_elapsed = time.perf_counter() - t0
    set_gbps = total_bytes / set_elapsed / 1e9
    print(f"    Valkey SET done: {set_elapsed:.1f}s", flush=True)

    # GET (same pattern as ValkeyConnector._get)
    print("    Valkey GET starting...", flush=True)

    async def do_get():
        results = []
        for key in keys:
            metadata_key, kv_key = get_keys(key)
            resp = await client.mget([metadata_key, kv_key])
            results.append(resp)
        return results

    t0 = time.perf_counter()
    get_results = asyncio.run_coroutine_threadsafe(
        do_get(), loop
    ).result(timeout=120)
    get_elapsed = time.perf_counter() - t0
    get_gbps = total_bytes / get_elapsed / 1e9
    print(f"    Valkey GET done: {get_elapsed:.1f}s", flush=True)

    # Verify
    mismatches = 0
    for i in range(num_keys):
        metadata_bytes, kv_bytes = get_results[i][0], get_results[i][1]
        if kv_bytes is None:
            mismatches += 1
            continue
        orig = memory_objs[i].byte_array
        if isinstance(orig, memoryview):
            orig = bytes(orig)
        if kv_bytes[:chunk_bytes] != orig[:chunk_bytes]:
            mismatches += 1

    asyncio.run_coroutine_threadsafe(client.close(), loop).result(timeout=5)

    return set_gbps, get_gbps, mismatches


# ── Benchmark: ValkeyConnector sync optimized ───────────────────────────────


def _mp_worker_init(host, port):
    """Called once per worker process at pool creation time."""
    import glide_sync as _gs
    global _glide_client
    config = _gs.GlideClientConfiguration(
        addresses=[_gs.NodeAddress(host, port)], request_timeout=120000,
    )
    _glide_client = _gs.GlideClient.create(config)


def _mp_worker_set(args):
    """Worker process: SET a slice of keys from shared memory."""
    from multiprocessing import shared_memory
    indices, shm_name, chunk_bytes, key_strs, metadata_bytes = args
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = memoryview(shm.buf)
    for i in indices:
        offset = i * chunk_bytes
        _glide_client.set(f"{key_strs[i]}:kv_bytes".encode(), buf[offset:offset + chunk_bytes])
        _glide_client.set(f"{key_strs[i]}:metadata".encode(), metadata_bytes)
    del buf
    shm.close()


def _mp_worker_get(args):
    """Worker process: GET a slice of keys into shared memory."""
    from multiprocessing import shared_memory
    from lmcache.v1.protocol import RemoteMetadata, init_remote_metadata_info
    indices, shm_name, chunk_bytes, key_strs, has_buffer_get, num_groups = args
    init_remote_metadata_info(num_groups)
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = memoryview(shm.buf)
    for i in indices:
        offset = i * chunk_bytes
        meta_bytes = _glide_client.get(f"{key_strs[i]}:metadata".encode())
        if meta_bytes is None:
            continue
        meta = RemoteMetadata.deserialize(memoryview(meta_bytes))
        target = buf[offset:offset + meta.length]
        if has_buffer_get:
            _glide_client.get(f"{key_strs[i]}:kv_bytes".encode(), buffer=target)
        else:
            kv_bytes = _glide_client.get(f"{key_strs[i]}:kv_bytes".encode())
            if kv_bytes is not None:
                target[:] = kv_bytes
        del target
    del buf
    shm.close()


def bench_valkey_sync_optimized(
    host, port, num_workers, keys, memory_objs
):
    """Benchmark GLIDE sync client with zero-copy SET + buffer GET.

    Uses N worker processes (not threads) to avoid GIL contention.
    Each process gets its own GLIDE sync client and Tokio runtime.
    Shared memory is used to pass KV data without copying across processes.

    Requires valkey-glide with PRs #5492 (memoryview SET) and #5493 (buffer GET).
    """
    import multiprocessing as mp
    from multiprocessing import shared_memory

    try:
        from glide_sync import GlideClient as SyncGlideClient
    except ImportError:
        print("  SKIPPED — glide_sync not available")
        return None, None, 0

    num_keys = len(keys)
    chunk_bytes = memory_objs[0].get_size()
    total_bytes = num_keys * chunk_bytes

    has_buffer_get = "buffer" in SyncGlideClient.get.__code__.co_varnames
    print(f"    buffer GET: {has_buffer_get}", flush=True)

    # Create shared memory for all KV data
    shm = shared_memory.SharedMemory(create=True, size=total_bytes)
    shm_buf = memoryview(shm.buf)

    # Copy source data into shared memory
    for i, mobj in enumerate(memory_objs):
        src = mobj.byte_array
        if isinstance(src, memoryview) and src.format != "B":
            src = src.cast("B")
        elif not isinstance(src, memoryview):
            src = memoryview(src)
        offset = i * chunk_bytes
        shm_buf[offset:offset + chunk_bytes] = src[:chunk_bytes]

    # Key strings for workers
    key_strs = [k.to_string() for k in keys]

    # Metadata bytes (same for all keys in this benchmark)
    from lmcache.v1.protocol import RemoteMetadata
    kv_shapes = memory_objs[0].get_shapes()
    kv_dtypes = memory_objs[0].get_dtypes()
    memory_format = memory_objs[0].get_memory_format()
    metadata_bytes = RemoteMetadata(
        chunk_bytes, kv_shapes, kv_dtypes, memory_format
    ).serialize()

    # Shared memory for GET results
    shm_get = shared_memory.SharedMemory(create=True, size=total_bytes)

    # Tile keys across workers
    def tile(n, num_tiles):
        size = (n + num_tiles - 1) // num_tiles
        return [list(range(i, min(i + size, n))) for i in range(0, n, size)]

    tiles = tile(num_keys, num_workers)

    # Create persistent pool (simulates production connector init)
    print("    Creating worker pool...", flush=True)
    pool = mp.Pool(num_workers, initializer=_mp_worker_init, initargs=(host, port))
    print(f"    {num_workers} worker processes ready", flush=True)

    num_groups = len(memory_objs[0].get_shapes())

    # ── SET ──
    print("    Flushing server...", flush=True)
    flush_server(host, port)
    print("    Sync SET starting...", flush=True)

    set_args = [
        (indices, shm.name, chunk_bytes, key_strs, metadata_bytes)
        for indices in tiles
    ]
    t0 = time.perf_counter()
    pool.map(_mp_worker_set, set_args)
    set_elapsed = time.perf_counter() - t0
    set_gbps = total_bytes / set_elapsed / 1e9
    print(f"    Sync SET done: {set_elapsed:.1f}s", flush=True)

    # ── GET ──
    print("    Sync GET starting...", flush=True)

    get_args = [
        (indices, shm_get.name, chunk_bytes, key_strs, has_buffer_get, num_groups)
        for indices in tiles
    ]
    t0 = time.perf_counter()
    pool.map(_mp_worker_get, get_args)
    get_elapsed = time.perf_counter() - t0
    get_gbps = total_bytes / get_elapsed / 1e9
    print(f"    Sync GET done: {get_elapsed:.1f}s", flush=True)

    pool.close()
    pool.join()

    # Verify: compare shm_get against shm (source data)
    mismatches = 0
    src_buf = memoryview(shm.buf)
    dst_buf = memoryview(shm_get.buf)
    for i in range(num_keys):
        offset = i * chunk_bytes
        if src_buf[offset:offset + chunk_bytes] != dst_buf[offset:offset + chunk_bytes]:
            mismatches += 1

    # Cleanup shared memory
    del src_buf, dst_buf, shm_buf
    shm_get.close()
    shm_get.unlink()
    shm.close()
    shm.unlink()

    label_suffix = "multiprocess, buffer GET" if has_buffer_get else "multiprocess, copy GET"
    return set_gbps, get_gbps, mismatches, label_suffix


# ── Main ─────────────────────────────────────────────────────────────────────


def run_benchmark(host, port, num_workers, model_params, num_keys):
    num_layers = model_params["num_layers"]
    chunk_tokens = model_params["chunk_tokens"]
    num_kv_heads = model_params["num_kv_heads"]
    head_dim = model_params["head_dim"]
    desc = model_params["desc"]

    # chunk_bytes = num_layers * 2 * chunk_tokens * num_kv_heads
    #               * head_dim * 2 (bf16)
    chunk_bytes = num_layers * 2 * chunk_tokens * num_kv_heads * head_dim * 2
    total_bytes = chunk_bytes * num_keys

    print(f"\n{'=' * 70}")
    print(f"LMCache Connector Benchmark: {desc}")
    print(f"Chunk: {chunk_bytes / 1e6:.0f} MB × {num_keys} keys "
          f"= {total_bytes / 1e9:.1f} GB")
    print(f"Workers: {num_workers}, Server: {host}:{port}")
    print(f"{'=' * 70}", flush=True)

    # Create shared infrastructure
    print("Creating metadata...", flush=True)
    metadata = make_metadata(num_layers, chunk_tokens, num_kv_heads, head_dim)

    # Initialize remote metadata format (needed by RemoteMetadata.serialize)
    from lmcache.v1.protocol import init_remote_metadata_info
    init_remote_metadata_info(len(metadata.get_shapes()))

    # RESP config (save_chunk_meta=False)
    resp_config = LMCacheEngineConfig.from_defaults(
        extra_config={
            "save_chunk_meta": False,
            "resp_num_threads": num_workers,
        }
    )

    # Valkey config (save_chunk_meta=True, the default)
    valkey_config = LMCacheEngineConfig.from_defaults()

    print(f"Allocating buffers...", flush=True)
    # Each benchmark gets its own allocator to avoid memory exhaustion
    # (connectors allocate new MemoryObjs internally on GET)
    def make_allocator():
        return TensorMemoryAllocator(
            torch.empty(total_bytes * 6, dtype=torch.uint8)
        )

    print("Creating backends...", flush=True)
    resp_backend = LocalCPUBackend(
        config=resp_config, metadata=metadata, memory_allocator=make_allocator()
    )
    valkey_backend = LocalCPUBackend(
        config=valkey_config, metadata=metadata, memory_allocator=make_allocator()
    )

    print("Starting event loop...", flush=True)
    loop, loop_thread = init_asyncio_loop()

    # Create keys and fill memory objects
    print(f"Allocating {num_keys} memory objects ({chunk_bytes/1e6:.0f} MB each)...",
          flush=True)
    keys = [make_key(i) for i in range(num_keys)]

    shapes = metadata.get_shapes()
    dtypes = metadata.get_dtypes()

    # Allocate memory objects for RESP
    resp_objs = []
    for i in range(num_keys):
        mobj = resp_backend.allocate(shapes, dtypes)
        fill_memory_obj(mobj, seed=42 + i)
        resp_objs.append(mobj)

    # Allocate memory objects for Valkey (same data)
    valkey_objs = []
    for i in range(num_keys):
        mobj = valkey_backend.allocate(shapes, dtypes)
        mobj.raw_data.copy_(resp_objs[i].raw_data)
        valkey_objs.append(mobj)

    print("Setup complete, starting benchmarks...", flush=True)

    results = {}

    # ── 1. RESPConnector ──
    print("\n[1] RESPConnector (C++ RESP client)...", flush=True)
    try:
        s, g, m = bench_resp_connector(
            host, port, num_workers, keys, resp_objs, loop
        )
        results["RESPConnector (C++ baseline)"] = (s, g)
        print(f"    SET: {s:.3f} GB/s  GET: {g:.3f} GB/s  "
              f"{'✓ verified' if m == 0 else f'✗ {m} mismatches'}", flush=True)
    except ImportError as e:
        print(f"    SKIPPED — {e}", flush=True)
    except Exception as e:
        print(f"    FAILED: {e}", flush=True)

    # ── 2. ValkeyConnector sync optimized ──
    print("\n[2] ValkeyConnector (sync GLIDE, zero-copy SET + buffer GET)...",
          flush=True)
    try:
        result = bench_valkey_sync_optimized(
            host, port, num_workers, keys, resp_objs
        )
        if result[0] is not None:
            s, g, m, label = result
            results[f"ValkeyConnector sync ({label})"] = (s, g)
            print(f"    SET: {s:.3f} GB/s  GET: {g:.3f} GB/s  "
                  f"{'✓ verified' if m == 0 else f'✗ {m} mismatches'}", flush=True)
    except Exception as e:
        import traceback
        print(f"    FAILED: {e}")
        traceback.print_exc()

    # ── 3. ValkeyConnector (async, as-is) ──
    print("\n[3] ValkeyConnector (async GLIDE, as-is)...", flush=True)
    try:
        s, g, m = bench_valkey_connector(
            host, port, keys, valkey_objs, loop
        )
        results["ValkeyConnector (async GLIDE)"] = (s, g)
        print(f"    SET: {s:.3f} GB/s  GET: {g:.3f} GB/s  "
              f"{'✓ verified' if m == 0 else f'✗ {m} mismatches'}", flush=True)
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"{'Mode':<50} {'SET (GB/s)':>10} {'GET (GB/s)':>10}")
    print(f"{'-' * 70}")
    for mode, (s, g) in results.items():
        print(f"{mode:<50} {s:>10.3f} {g:>10.3f}")
    print(f"{'=' * 70}")

    # Cleanup
    close_asyncio_loop(loop, loop_thread)
    resp_backend.close()
    valkey_backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark LMCache connector implementations"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-keys", type=int, default=None,
                        help="Override number of keys (chunks)")
    parser.add_argument(
        "--preset", default="llama8b-8k",
        help="Model-context preset: llama8b-8k, llama8b-64k, "
             "llama70b-8k, llama70b-64k",
    )
    args = parser.parse_args()

    model_params, preset_num_keys = get_preset_params(args.preset)
    num_keys = args.num_keys if args.num_keys is not None else preset_num_keys

    run_benchmark(args.host, args.port, args.num_workers, model_params, num_keys)
