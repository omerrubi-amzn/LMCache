# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``GlideBatchClient`` — the pure-Python batch/eventfd shim that
adapts the GLIDE thread-worker pool to the native-client interface consumed by
``NativeConnectorL2Adapter``.

These tests use a fake in-memory pool, so they need neither ``glide_sync`` nor a
running Valkey server.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
import select

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.glide_batch_client import GlideBatchClient


class FakePool:
    """In-memory stand-in for ``_ThreadWorkerPool``.

    Exposes the same ``submit_*`` methods returning ``concurrent.futures.Future``
    and a ``close()``. ``fail_keys`` forces the SET of those keys to raise, to
    exercise the store error path.
    """

    def __init__(self, fail_keys=frozenset()):
        self.store: dict[str, bytes] = {}
        self.fail_keys = set(fail_keys)
        self._ex = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fakepool")

    def submit_set(self, key: str, data) -> Future:
        return self._ex.submit(self._set, key, bytes(data))

    def _set(self, key: str, data: bytes) -> None:
        if key in self.fail_keys:
            raise RuntimeError(f"forced set failure for {key}")
        self.store[key] = data

    def submit_get_into(self, key: str, buf: memoryview) -> Future:
        return self._ex.submit(self._get_into, key, buf)

    def _get_into(self, key: str, buf: memoryview) -> bool:
        data = self.store.get(key)
        if data is None:
            return False
        if len(data) != buf.nbytes:
            return False
        buf[: len(data)] = data
        return True

    def submit_exists(self, key: str) -> Future:
        return self._ex.submit(lambda: key in self.store)

    def submit_delete(self, key: str) -> Future:
        return self._ex.submit(self._delete, key)

    def _delete(self, key: str) -> bool:
        existed = key in self.store
        self.store.pop(key, None)
        return existed

    def close(self) -> None:
        self._ex.shutdown(wait=True)


def _wait_drain(client: GlideBatchClient, timeout_s: float = 5.0):
    """Poll the client's eventfd until readable, then drain and return results."""
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    events = poller.poll(int(timeout_s * 1000))
    assert events, "eventfd was not signaled within timeout"
    return client.drain_completions()


def _mv(data: bytes) -> memoryview:
    return memoryview(bytearray(data))


@pytest.fixture
def client():
    c = GlideBatchClient(FakePool())
    yield c
    c.close()


def test_set_then_get_round_trip(client):
    keys = ["k0", "k1", "k2"]
    values = [b"a" * 16, b"b" * 32, b"c" * 8]

    fid_set = client.submit_batch_set(keys, [memoryview(v) for v in values])
    comps = _wait_drain(client)
    assert len(comps) == 1
    got_fid, ok, err, bools = comps[0]
    assert got_fid == fid_set
    assert ok is True and err == ""
    assert bools == [True, True, True]

    # Load into correctly-sized destination buffers.
    dsts = [_mv(b"\x00" * len(v)) for v in values]
    fid_get = client.submit_batch_get(keys, dsts)
    comps = _wait_drain(client)
    got_fid, ok, err, bools = comps[0]
    assert got_fid == fid_get
    assert ok is True
    assert bools == [True, True, True]
    for dst, v in zip(dsts, values, strict=True):
        assert bytes(dst) == v


def test_get_miss_and_size_mismatch(client):
    client.submit_batch_set(["present"], [memoryview(b"x" * 10)])
    _wait_drain(client)

    # present (right size) -> True; absent -> False; present-but-wrong-size -> False
    dsts = [_mv(b"\x00" * 10), _mv(b"\x00" * 10), _mv(b"\x00" * 4)]
    client.submit_batch_get(["present", "absent", "present"], dsts)
    comps = _wait_drain(client)
    _, ok, _, bools = comps[0]
    assert ok is True
    assert bools == [True, False, False]


def test_exists(client):
    client.submit_batch_set(["e1", "e2"], [memoryview(b"1"), memoryview(b"2")])
    _wait_drain(client)

    client.submit_batch_exists(["e1", "missing", "e2"])
    comps = _wait_drain(client)
    _, ok, _, bools = comps[0]
    assert ok is True
    assert bools == [True, False, True]


def test_delete(client):
    client.submit_batch_set(["d1", "d2"], [memoryview(b"1"), memoryview(b"2")])
    _wait_drain(client)

    client.submit_batch_delete(["d1", "nope", "d2"])
    comps = _wait_drain(client)
    _, ok, _, bools = comps[0]
    assert ok is True
    assert bools == [True, False, True]

    # Both deleted keys should now be gone.
    client.submit_batch_exists(["d1", "d2"])
    comps = _wait_drain(client)
    _, _, _, bools = comps[0]
    assert bools == [False, False]


def test_store_error_marks_task_failed():
    c = GlideBatchClient(FakePool(fail_keys={"bad"}))
    try:
        c.submit_batch_set(["ok1", "bad", "ok2"], [memoryview(b"x")] * 3)
        comps = _wait_drain(c)
        _, ok, err, _ = comps[0]
        assert ok is False
        assert "forced set failure" in err
    finally:
        c.close()


def test_future_ids_are_unique_and_monotonic(client):
    fid1 = client.submit_batch_exists(["a"])
    _wait_drain(client)
    fid2 = client.submit_batch_exists(["b"])
    _wait_drain(client)
    assert fid2 == fid1 + 1


def test_empty_batch_completes_immediately(client):
    fid = client.submit_batch_exists([])
    comps = _wait_drain(client)
    assert comps == [(fid, True, "", [])]


def test_launch_failure_finalizes_batch_without_hanging():
    """If the pool raises mid fan-out, the batch must still complete (ok=False)."""

    class FlakyPool(FakePool):
        def __init__(self):
            super().__init__()
            self._n = 0

        def submit_set(self, key, data):
            self._n += 1
            if self._n == 2:
                raise RuntimeError("cannot schedule new futures after shutdown")
            return super().submit_set(key, data)

    c = GlideBatchClient(FlakyPool())
    try:
        c.submit_batch_set(["a", "b", "c"], [memoryview(b"x")] * 3)
        comps = _wait_drain(c)
        assert len(comps) == 1
        _, ok, err, _ = comps[0]
        assert ok is False
        assert "cannot schedule" in err
    finally:
        c.close()


def test_drain_consumes_eventfd(client):
    client.submit_batch_exists(["x"])
    comps = _wait_drain(client)
    assert len(comps) == 1
    # After draining, the fd must no longer be readable (no busy-spin).
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    assert poller.poll(200) == []
    # And a subsequent drain returns nothing.
    assert client.drain_completions() == []
