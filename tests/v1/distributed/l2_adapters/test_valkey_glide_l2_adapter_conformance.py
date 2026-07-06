# SPDX-License-Identifier: Apache-2.0
"""
Network-free full-stack conformance test for the valkey_glide L2 adapter.

Drives the **real** ``NativeConnectorL2Adapter`` + ``GlideBatchClient`` through
the public L2 adapter interface (store -> lookup+lock -> load -> delete + usage),
with the GLIDE ``_ThreadWorkerPool`` replaced by an in-memory fake. This
validates the full MP-mode wiring (eventfds, bitmaps, byte accounting, delete)
without needing ``glide_sync`` or a running Valkey server.

A separate integration test (test_valkey_glide_l2_adapter_integration.py)
exercises the same paths against a real Valkey.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
import select
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


class FakeGlidePool:
    """In-memory stand-in for ``_ThreadWorkerPool`` accepting its full kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.store: dict[bytes, bytes] = {}
        self._ex = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fakeglide")

    def submit_set(self, key: str, data) -> Future:
        return self._ex.submit(self._set, key.encode(), bytes(data))

    def _set(self, key: bytes, data: bytes) -> None:
        self.store[key] = data

    def submit_get_into(self, key: str, buf: memoryview) -> Future:
        return self._ex.submit(self._get_into, key.encode(), buf)

    def _get_into(self, key: bytes, buf: memoryview) -> bool:
        if buf.format != "B":
            buf = buf.cast("B")
        data = self.store.get(key)
        if data is None or len(data) != buf.nbytes:
            return False
        buf[: len(data)] = data
        return True

    def submit_exists(self, key: str) -> Future:
        return self._ex.submit(lambda: key.encode() in self.store)

    def submit_delete(self, key: str) -> Future:
        return self._ex.submit(self._delete, key.encode())

    def _delete(self, key: bytes) -> bool:
        existed = key in self.store
        self.store.pop(key, None)
        return existed

    def close(self) -> None:
        self._ex.shutdown(wait=True)


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def _obj(size: int = 128, fill: float = 1.0) -> TensorMemoryObj:
    raw = torch.empty(size, dtype=torch.float32)
    raw.fill_(fill)
    meta = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _wait(fd: int, timeout: float = 10.0) -> bool:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    if poll.poll(timeout * 1000):
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        return True
    return False


@pytest.fixture
def adapter(monkeypatch):
    # Make the factory's lazy `import glide_sync` succeed, and swap the pool.
    monkeypatch.setitem(sys.modules, "glide_sync", type(sys)("glide_sync"))
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connector.glide_pool._ThreadWorkerPool",
        FakeGlidePool,
    )
    # First Party
    from lmcache.v1.distributed.l2_adapters import create_l2_adapter
    from lmcache.v1.distributed.l2_adapters.valkey_glide_l2_adapter import (
        ValkeyGlideL2AdapterConfig,
    )

    cfg = ValkeyGlideL2AdapterConfig.from_dict(
        {"type": "valkey_glide", "host": "fake", "port": 6379, "max_capacity_gb": 1}
    )
    a = create_l2_adapter(cfg)
    yield a
    a.close()


def test_event_fds_distinct(adapter):
    fds = {
        adapter.get_store_event_fd(),
        adapter.get_lookup_and_lock_event_fd(),
        adapter.get_load_event_fd(),
    }
    assert len(fds) == 3


def test_store_lookup_load_roundtrip(adapter):
    keys = [_key(i) for i in range(6)]
    store_objs = [_obj(128, float(i + 1)) for i in range(6)]
    load_objs = [_obj(128, 0.0) for _ in range(6)]

    tid = adapter.submit_store_task(keys, store_objs)
    assert _wait(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[tid].is_successful()

    tid = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
    assert _wait(adapter.get_lookup_and_lock_event_fd())
    bm = adapter.query_lookup_and_lock_result(tid)
    assert bm is not None and all(bm.test(i) for i in range(6))

    tid = adapter.submit_load_task(keys, load_objs)
    assert _wait(adapter.get_load_event_fd())
    bm = adapter.query_load_result(tid)
    assert all(bm.test(i) for i in range(6))
    for lo, so in zip(load_objs, store_objs, strict=True):
        assert torch.allclose(lo.tensor, so.tensor)

    adapter.submit_unlock(keys)


def test_lookup_mixed_present_missing(adapter):
    present = [_key(i) for i in range(3)]
    objs = [_obj(64, float(i)) for i in range(3)]
    adapter.submit_store_task(present, objs)
    assert _wait(adapter.get_store_event_fd())
    adapter.pop_completed_store_tasks()

    all_keys = present + [_key(500), _key(501)]
    tid = adapter.submit_lookup_and_lock_task(all_keys, _EMPTY_LAYOUT)
    assert _wait(adapter.get_lookup_and_lock_event_fd())
    bm = adapter.query_lookup_and_lock_result(tid)
    assert all(bm.test(i) for i in range(3))
    assert not bm.test(3) and not bm.test(4)
    adapter.submit_unlock(present)


def test_delete_and_usage_accounting(adapter):
    keys = [_key(i) for i in range(4)]
    objs = [_obj(256, float(i + 1)) for i in range(4)]

    tid = adapter.submit_store_task(keys, objs)
    assert _wait(adapter.get_store_event_fd())
    assert adapter.pop_completed_store_tasks()[tid].is_successful()

    usage = adapter.get_usage()
    assert usage.total_bytes_used == 4 * 256 * 4  # 4 objs * 256 floats * 4 bytes

    # delete() is synchronous in NativeConnectorL2Adapter.
    adapter.delete(keys)
    usage2 = adapter.get_usage()
    assert usage2.total_bytes_used == 0
