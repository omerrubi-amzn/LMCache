# SPDX-License-Identifier: Apache-2.0
"""
Integration test for the valkey_glide L2 adapter in MP mode.

Requires a running Valkey/Redis server AND the ``glide_sync`` client. Skipped if
either is unavailable (mirrors test_resp_l2_adapter_integration.py).

Server location via env: VALKEY_HOST (default localhost), VALKEY_PORT
(default 6399).
"""

# Standard
import os
import select
import socket

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

VALKEY_HOST = os.environ.get("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.environ.get("VALKEY_PORT", "6399"))


def _valkey_available() -> bool:
    try:
        with socket.create_connection((VALKEY_HOST, VALKEY_PORT), timeout=3):
            return True
    except OSError:
        return False


def _glide_available() -> bool:
    try:
        import glide_sync  # noqa: F401

        return True
    except ImportError:
        return False


requires_valkey = pytest.mark.skipif(
    not _valkey_available(),
    reason=f"Valkey not reachable at {VALKEY_HOST}:{VALKEY_PORT}",
)
requires_glide = pytest.mark.skipif(
    not _glide_available(),
    reason="glide_sync client not available",
)


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="valkey_glide_it",
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


@requires_valkey
@requires_glide
class TestValkeyGlideL2AdapterIntegration:
    """End-to-end MP-mode tests against a real Valkey server via GLIDE."""

    @pytest.fixture(autouse=True)
    def setup_adapter(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter
        from lmcache.v1.distributed.l2_adapters.valkey_glide_l2_adapter import (
            ValkeyGlideL2AdapterConfig,
        )

        cfg = ValkeyGlideL2AdapterConfig.from_dict(
            {
                "type": "valkey_glide",
                "host": VALKEY_HOST,
                "port": VALKEY_PORT,
                "num_workers": 4,
                "max_capacity_gb": 1,
            }
        )
        self.adapter = create_l2_adapter(cfg)
        yield
        self.adapter.close()

    def test_event_fds_distinct(self):
        fds = {
            self.adapter.get_store_event_fd(),
            self.adapter.get_lookup_and_lock_event_fd(),
            self.adapter.get_load_event_fd(),
        }
        assert len(fds) == 3

    def test_store_lookup_load_workflow(self):
        keys = [_key(i) for i in range(8)]
        store_objs = [_obj(256, float(i + 1)) for i in range(8)]
        load_objs = [_obj(256, 0.0) for _ in range(8)]

        tid = self.adapter.submit_store_task(keys, store_objs)
        assert _wait(self.adapter.get_store_event_fd())
        assert self.adapter.pop_completed_store_tasks()[tid].is_successful()

        tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert _wait(self.adapter.get_lookup_and_lock_event_fd())
        bm = self.adapter.query_lookup_and_lock_result(tid)
        assert bm is not None and all(bm.test(i) for i in range(8))

        tid = self.adapter.submit_load_task(keys, load_objs)
        assert _wait(self.adapter.get_load_event_fd())
        bm = self.adapter.query_load_result(tid)
        assert all(bm.test(i) for i in range(8))
        for lo, so in zip(load_objs, store_objs, strict=True):
            assert torch.allclose(lo.tensor, so.tensor)

        self.adapter.submit_unlock(keys)

    def test_lookup_missing_keys(self):
        keys = [_key(i + 9000) for i in range(4)]
        tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert _wait(self.adapter.get_lookup_and_lock_event_fd())
        bm = self.adapter.query_lookup_and_lock_result(tid)
        assert bm is not None
        assert not any(bm.test(i) for i in range(4))

    def test_delete_removes_keys(self):
        keys = [_key(i + 100) for i in range(3)]
        objs = [_obj(128, float(i + 1)) for i in range(3)]
        tid = self.adapter.submit_store_task(keys, objs)
        assert _wait(self.adapter.get_store_event_fd())
        self.adapter.pop_completed_store_tasks()

        self.adapter.delete(keys)

        tid = self.adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
        assert _wait(self.adapter.get_lookup_and_lock_event_fd())
        bm = self.adapter.query_lookup_and_lock_result(tid)
        assert not any(bm.test(i) for i in range(3))

    def test_factory_creates_adapter(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter
        from lmcache.v1.distributed.l2_adapters.valkey_glide_l2_adapter import (
            ValkeyGlideL2AdapterConfig,
        )

        cfg = ValkeyGlideL2AdapterConfig(
            host=VALKEY_HOST, port=VALKEY_PORT, num_workers=2
        )
        adapter = create_l2_adapter(cfg)
        try:
            assert adapter.get_store_event_fd() >= 0
            assert adapter.get_lookup_and_lock_event_fd() >= 0
            assert adapter.get_load_event_fd() >= 0
        finally:
            adapter.close()
