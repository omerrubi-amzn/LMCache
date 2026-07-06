#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
Performance benchmark for the MP-mode valkey_glide L2 adapter.

Drives the assembled adapter (GlideBatchClient behind NativeConnectorL2Adapter)
against a real Valkey server through the public L2 interface and reports store /
load throughput (ops/s, MiB/s) and load latency percentiles (p50/p99) across a
matrix of value sizes, batch sizes, and worker counts.

Usage:
    VALKEY_HOST=localhost VALKEY_PORT=6399 \\
      python benchmarks/valkey_glide_mp_bench.py [--out results.md]

Requires ``glide_sync`` and a reachable Valkey server. On stock valkey-glide,
run with the local async->sync shim on PYTHONPATH (see the effort harness).
"""

# Standard
from statistics import median
import argparse
import os
import select
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.valkey_glide_l2_adapter import (
    ValkeyGlideL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY = MemoryLayoutDesc(shapes=[], dtypes=[])
HOST = os.environ.get("VALKEY_HOST", "localhost")
PORT = int(os.environ.get("VALKEY_PORT", "6399"))

VALUE_SIZES = [16 * 1024, 64 * 1024, 256 * 1024]  # bytes
BATCH_SIZES = [1, 32, 128]
WORKER_COUNTS = [4, 8]
TARGET_BYTES = 48 * 1024 * 1024  # ~48 MiB of traffic per config


def _key(i: int, salt: str) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i),
        model_name=f"bench_{salt}",
        kv_rank=0,
    )


def _obj(nbytes: int, fill: float) -> TensorMemoryObj:
    n = nbytes // 4
    raw = torch.empty(n, dtype=torch.float32)
    raw.fill_(fill)
    meta = MemoryObjMetadata(
        shape=torch.Size([n]),
        dtype=torch.float32,
        address=0,
        phy_size=nbytes,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _wait(fd: int, timeout: float = 30.0) -> None:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    assert poll.poll(timeout * 1000), "timeout waiting for event fd"
    try:
        consume_fd(fd)
    except BlockingIOError:
        pass


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def bench_config(value_size: int, batch: int, workers: int) -> dict:
    n = max(batch, TARGET_BYTES // value_size)
    n = (n // batch) * batch  # multiple of batch
    salt = f"{value_size}_{batch}_{workers}"

    cfg = ValkeyGlideL2AdapterConfig.from_dict(
        {"type": "valkey_glide", "host": HOST, "port": PORT, "num_workers": workers}
    )
    adapter = create_l2_adapter(cfg)
    store_fd = adapter.get_store_event_fd()
    load_fd = adapter.get_load_event_fd()
    try:
        keys = [_key(i, salt) for i in range(n)]
        store_objs = [_obj(value_size, float(1 + (i % 7))) for i in range(n)]

        # ---- STORE ----
        t0 = time.perf_counter()
        for kb, ob in zip(
            _chunks(keys, batch), _chunks(store_objs, batch), strict=True
        ):
            tid = adapter.submit_store_task(list(kb), list(ob))
            _wait(store_fd)
            assert adapter.pop_completed_store_tasks()[tid].is_successful()
        store_dt = time.perf_counter() - t0

        # ---- LOAD ---- (fresh buffers; record per-batch latency)
        load_objs = [_obj(value_size, 0.0) for _ in range(n)]
        latencies = []
        t0 = time.perf_counter()
        for kb, ob in zip(_chunks(keys, batch), _chunks(load_objs, batch), strict=True):
            b0 = time.perf_counter()
            tid = adapter.submit_load_task(list(kb), list(ob))
            _wait(load_fd)
            bm = adapter.query_load_result(tid)
            latencies.append((time.perf_counter() - b0) * 1000.0)
            assert bm is not None
        load_dt = time.perf_counter() - t0

        # cleanup keys from the server
        adapter.delete(keys)

        total_mib = n * value_size / (1024 * 1024)
        latencies.sort()
        p50 = median(latencies)
        p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]
        return {
            "value_kib": value_size // 1024,
            "batch": batch,
            "workers": workers,
            "n": n,
            "store_ops_s": n / store_dt,
            "store_mib_s": total_mib / store_dt,
            "load_ops_s": n / load_dt,
            "load_mib_s": total_mib / load_dt,
            "load_p50_ms": p50,
            "load_p99_ms": p99,
        }
    finally:
        adapter.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write a markdown report to this path")
    args = ap.parse_args()

    rows = []
    for w in WORKER_COUNTS:
        for vs in VALUE_SIZES:
            for b in BATCH_SIZES:
                r = bench_config(vs, b, w)
                rows.append(r)
                print(
                    f"workers={r['workers']} "
                    f"value={r['value_kib']}KiB "
                    f"batch={r['batch']:>3} | "
                    f"store {r['store_ops_s']:8.0f} ops/s "
                    f"{r['store_mib_s']:7.1f} MiB/s | "
                    f"load {r['load_ops_s']:8.0f} ops/s "
                    f"{r['load_mib_s']:7.1f} MiB/s | "
                    f"p50 {r['load_p50_ms']:6.2f}ms "
                    f"p99 {r['load_p99_ms']:6.2f}ms"
                )

    if args.out:
        lines = [
            "# valkey_glide MP adapter — benchmark results",
            "",
            f"Server: {HOST}:{PORT} | torch {torch.__version__}",
            "",
            "| workers | value KiB | batch | store ops/s | store MiB/s | "
            "load ops/s | load MiB/s | load p50 ms | load p99 ms |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['workers']} | {r['value_kib']} | {r['batch']} | "
                f"{r['store_ops_s']:.0f} | {r['store_mib_s']:.1f} | "
                f"{r['load_ops_s']:.0f} | {r['load_mib_s']:.1f} | "
                f"{r['load_p50_ms']:.2f} | {r['load_p99_ms']:.2f} |"
            )
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
