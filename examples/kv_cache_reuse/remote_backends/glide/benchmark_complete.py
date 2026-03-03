#!/usr/bin/env python3
"""Complete LMCache benchmark: GLIDE sync (all modes) + C++ RESP baseline."""
import argparse, asyncio, time, os, sys, concurrent.futures

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from glide_sync import GlideClientConfiguration, NodeAddress
from glide_sync.glide_client import GlideClient

FAKE_META = b"\x00" * 28


def bench_glide(host, port, chunk_size, num_keys, num_workers, buffers, total_bytes):
    cfg = GlideClientConfiguration([NodeAddress(host, port)], request_timeout=120000)
    clients = [GlideClient.create(cfg) for _ in range(num_workers)]
    clients[0].custom_command(["FLUSHALL"])
    results = {}

    # SET: bytes() copy
    clients[0].custom_command(["FLUSHALL"])
    def set_bytes(i):
        c = clients[i % num_workers]
        c.set(f"a:{i}:kv".encode(), bytes(buffers[i]))
        c.set(f"a:{i}:meta".encode(), FAKE_META)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(set_bytes, range(num_keys)))
    results["SET: GLIDE sync bytes() copy"] = total_bytes / (time.perf_counter() - t0) / 1e9

    # SET: memoryview zero-copy
    clients[0].custom_command(["FLUSHALL"])
    def set_mv(i):
        c = clients[i % num_workers]
        c.set(f"b:{i}:kv".encode(), buffers[i])
        c.set(f"b:{i}:meta".encode(), FAKE_META)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(set_mv, range(num_keys)))
    results["SET: GLIDE sync memoryview (zero-copy)"] = total_bytes / (time.perf_counter() - t0) / 1e9

    # Write keys for GET
    clients[0].custom_command(["FLUSHALL"])
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(set_mv, range(num_keys)))

    # GET: regular
    tgt_a = [bytearray(chunk_size) for _ in range(num_keys)]
    def get_reg(i):
        c = clients[i % num_workers]
        r = c.get(f"b:{i}:kv".encode())
        memoryview(tgt_a[i])[:len(r)] = r
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(get_reg, range(num_keys)))
    results["GET: GLIDE sync get() + copy"] = total_bytes / (time.perf_counter() - t0) / 1e9

    # GET: get_into_buffer
    tgt_b = [bytearray(chunk_size) for _ in range(num_keys)]
    def get_buf(i):
        c = clients[i % num_workers]
        c.get_into_buffer(f"b:{i}:kv".encode(), memoryview(tgt_b[i]))
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(get_buf, range(num_keys)))
    results["GET: GLIDE sync get_into_buffer (zero-alloc)"] = total_bytes / (time.perf_counter() - t0) / 1e9

    assert tgt_a[0] == tgt_b[0], "Data mismatch!"
    for c in clients:
        c.close()
    return results


async def bench_resp(host, port, chunk_size, num_keys, num_workers, buffers, total_bytes):
    import lmcache_redis

    class RESPClient:
        def __init__(self, h, p, n):
            self._c = lmcache_redis.LMCacheRedisClient(h, p, n, "", "")
            self._loop = asyncio.get_running_loop()
            self._pending = {}
            self._loop.add_reader(self._c.event_fd(), self._drain)
        def _drain(self):
            for fid, ok, err, _ in self._c.drain_completions():
                fut = self._pending.pop(fid, None)
                if fut and not fut.done():
                    fut.set_result(None) if ok else fut.set_exception(RuntimeError(err))
        def _submit(self, fn, *a):
            fut = self._loop.create_future()
            self._pending[fn(*a)] = fut
            return fut
        async def batch_set(self, keys, mvs):
            await self._submit(self._c.submit_batch_set, keys, mvs)
        async def batch_get(self, keys, mvs):
            await self._submit(self._c.submit_batch_get, keys, mvs)
        def close(self):
            self._loop.remove_reader(self._c.event_fd())
            self._c.close()

    client = RESPClient(host, port, num_workers)
    keys = [f"resp:{i}" for i in range(num_keys)]
    results = {}

    # SET
    t0 = time.perf_counter()
    await client.batch_set(keys, [buffers[i] for i in range(num_keys)])
    results["SET: C++ RESP baseline"] = total_bytes / (time.perf_counter() - t0) / 1e9

    # GET
    read_bufs = [memoryview(bytearray(chunk_size)) for _ in range(num_keys)]
    t0 = time.perf_counter()
    await client.batch_get(keys, read_bufs)
    results["GET: C++ RESP baseline"] = total_bytes / (time.perf_counter() - t0) / 1e9

    client.close()
    return results


async def run_all(host, port, chunk_mb, num_keys, num_workers):
    chunk_size = int(chunk_mb * 1024 * 1024)
    total_bytes = chunk_size * num_keys
    data = os.urandom(chunk_size)
    buffers = [memoryview(bytearray(data)) for _ in range(num_keys)]

    # RESP baseline
    resp = await bench_resp(host, port, chunk_size, num_keys, num_workers, buffers, total_bytes)

    # GLIDE benchmarks
    glide = bench_glide(host, port, chunk_size, num_keys, num_workers, buffers, total_bytes)

    # Print
    all_results = {**resp, **glide}
    print(f"\n{'='*66}")
    print(f"LMCache Complete Benchmark: {chunk_mb:.0f} MB x {num_keys} keys = {total_bytes/1e9:.1f} GB")
    print(f"Workers: {num_workers}, Server: {host}:{port}")
    print(f"{'='*66}")
    print(f"{'Mode':<50} {'Throughput':>12}")
    print(f"{'-'*66}")
    for mode, gbps in all_results.items():
        print(f"{mode:<50} {gbps:>10.3f} GB/s")
    print(f"{'='*66}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--chunk-mb", type=float, default=32)
    p.add_argument("--num-keys", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    asyncio.run(run_all(args.host, args.port, args.chunk_mb, args.num_keys, args.num_workers))
