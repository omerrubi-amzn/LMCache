# SPDX-License-Identifier: Apache-2.0
"""
Standalone RESP benchmark using the compiled lmcache_redis C++ module directly.
Matches the Glide benchmark params for apples-to-apples comparison.
"""

import argparse
import asyncio
import os
import sys
import time

# Add repo root so we can import the compiled module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
import lmcache_redis


class RESPClientSimple:
    """Minimal async wrapper around LMCacheRedisClient for benchmarking."""

    def __init__(self, host, port, num_workers, username="", password=""):
        self._client = lmcache_redis.LMCacheRedisClient(
            host, port, num_workers, username, password
        )
        self._loop = asyncio.get_running_loop()
        self._pending = {}
        self._loop.add_reader(self._client.event_fd(), self._on_readable)

    def _on_readable(self):
        for future_id, ok, error, result_bools in self._client.drain_completions():
            fut = self._pending.pop(future_id, None)
            if fut and not fut.done():
                if ok:
                    fut.set_result(result_bools)
                else:
                    fut.set_exception(RuntimeError(error))

    def _submit(self, submit_fn, *args):
        fut = self._loop.create_future()
        future_id = submit_fn(*args)
        self._pending[future_id] = fut
        return fut

    async def batch_set(self, keys, memviews):
        await self._submit(self._client.submit_batch_set, keys, memviews)

    async def batch_get(self, keys, memviews):
        await self._submit(self._client.submit_batch_get, keys, memviews)

    def close(self):
        self._loop.remove_reader(self._client.event_fd())
        self._client.close()


async def run_benchmark(host, port, chunk_mb, num_workers, num_keys, username, password):
    chunk_bytes = int(chunk_mb * 1024 * 1024)

    client = RESPClientSimple(host, port, num_workers, username, password)

    print("RESP Client Benchmark (C++ lmcache_redis)")
    print(f"Server: {host}:{port}, Workers: {num_workers}")
    print(f"Chunk size: {chunk_bytes / 1024:.0f}KB, Keys: {num_keys}")
    print(f"Total data: {num_keys * chunk_mb:.0f}MB")
    print("-" * 60)

    print("Preparing buffers...")
    keys = [f"bench:resp:{i}" for i in range(num_keys)]
    buffers = [bytearray(os.urandom(chunk_bytes)) for _ in range(num_keys)]

    total_bytes = num_keys * chunk_bytes

    # Batch SET
    t0 = time.perf_counter()
    await client.batch_set(keys, [memoryview(b) for b in buffers])
    elapsed_set = time.perf_counter() - t0
    tp_set = total_bytes / elapsed_set / (1024**3)

    # Batch GET
    read_bufs = [bytearray(chunk_bytes) for _ in range(num_keys)]
    t0 = time.perf_counter()
    await client.batch_get(keys, [memoryview(b) for b in read_bufs])
    elapsed_get = time.perf_counter() - t0
    tp_get = total_bytes / elapsed_get / (1024**3)

    # Verify
    mismatches = sum(1 for i in range(num_keys) if read_bufs[i] != buffers[i])

    print(f"\n  SET: {tp_set:6.2f} GB/s ({elapsed_set:.3f}s)")
    print(f"  GET: {tp_get:6.2f} GB/s ({elapsed_get:.3f}s)")
    if mismatches:
        print(f"  WARNING: {mismatches}/{num_keys} keys had data mismatch")
    else:
        print("  Data verified OK")

    client.close()
    print("-" * 60)
    print("Done.")


# ── Model presets ────────────────────────────────────────────────────────────

MODEL_PRESETS = {
    "llama8b":  (32 * 2 * 256 * 8 * 128 * 2 / (1024**2),  "Llama-3.1-8B"),
    "llama70b": (80 * 2 * 256 * 8 * 128 * 2 / (1024**2),  "Llama-3.1-70B"),
}

CONTEXT_PRESETS = {"8k": 32, "64k": 256}


def apply_preset(args):
    if not args.preset:
        return
    parts = args.preset.split("-", 1)
    model_key = parts[0]
    ctx_key = parts[1] if len(parts) > 1 else None
    if model_key not in MODEL_PRESETS:
        raise SystemExit(
            f"Unknown model preset '{model_key}'. "
            f"Available: {', '.join(MODEL_PRESETS)}"
        )
    chunk_mb, desc = MODEL_PRESETS[model_key]
    if args.chunk_mb == 4.0:
        args.chunk_mb = chunk_mb
    if ctx_key:
        if ctx_key not in CONTEXT_PRESETS:
            raise SystemExit(
                f"Unknown context preset '{ctx_key}'. "
                f"Available: {', '.join(CONTEXT_PRESETS)}"
            )
        if args.num_keys == 500:
            args.num_keys = CONTEXT_PRESETS[ctx_key]
    print(f"Preset: {desc}, chunk={args.chunk_mb:.1f}MB, keys={args.num_keys}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RESP C++ client")
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
        help="Model-context preset, e.g. llama8b-8k, llama70b-64k.",
    )
    args = parser.parse_args()
    apply_preset(args)
    kwargs = vars(args)
    kwargs.pop("preset")
    asyncio.run(run_benchmark(**kwargs))
