# SPDX-License-Identifier: Apache-2.0
"""
GlideBatchClient — a pure-Python "native-like" batch client backed by the GLIDE
sync thread-worker pool.

This is the mirror image of ``storage_backend/native_clients/connector_client_base.py``:
where ``ConnectorClientBase`` adapts a native (C++) batch/eventfd client *down*
to asyncio coroutines for the non-MP connector, ``GlideBatchClient`` adapts the
pure-Python GLIDE ``_ThreadWorkerPool`` (``concurrent.futures``) *up* to the
duck-typed batch/eventfd interface expected by ``NativeConnectorL2Adapter``:

    event_fd() -> int
    submit_batch_get(keys, memviews) -> future_id
    submit_batch_set(keys, memviews) -> future_id
    submit_batch_exists(keys) -> future_id
    submit_batch_delete(keys) -> future_id
    drain_completions() -> list[(future_id, ok, err, result_bools)]
    close()

With this shim, ``NativeConnectorL2Adapter`` provides — for free — the three L2
eventfds, the background demux thread, client-side lock refcounting, byte
accounting, bitmap results, ``delete()``, ``get_usage()`` and ``report_status()``.

Completion model: a batch of N keys fans out to N pool futures. When the **last**
future in the batch resolves, the aggregated result
``(future_id, ok, err, result_bools)`` is enqueued and the single eventfd is
signaled once (one wakeup per batch, not per key).

Per-op result semantics (consumed by ``NativeConnectorL2Adapter._demux_loop``):
- set:    ``ok`` = every key stored without error (store error handling is
          coarse-grained at the task level); ``result_bools`` unused for stores.
- get:    ``ok`` = True; ``result_bools[i]`` = value found and size-matched.
- exists: ``ok`` = True; ``result_bools[i]`` = key present.
- delete: ``ok`` = True; ``result_bools[i]`` = key existed and was removed.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from concurrent.futures import Future
from typing import Optional
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import create_event_notifier
from lmcache.v1.storage_backend.connector.glide_pool import _ThreadWorkerPool

logger = init_logger(__name__)


# Operation tags for the pending-batch state.
_OP_SET = "set"
_OP_GET = "get"
_OP_EXISTS = "exists"
_OP_DELETE = "delete"


class _PendingBatch:
    """Mutable per-batch aggregation state.

    Tracks how many of the batch's per-key futures are still outstanding and
    accumulates the per-key boolean results in submit order. ``ok`` stays True
    unless a key in a *store* batch raised (see module docstring for the
    per-op contract).
    """

    __slots__ = ("op_type", "results", "remaining", "ok", "err")

    def __init__(self, op_type: str, num_keys: int) -> None:
        self.op_type = op_type
        self.results: list[bool] = [False] * num_keys
        self.remaining = num_keys
        self.ok = True
        self.err = ""


class GlideBatchClient:
    """Adapts a GLIDE ``_ThreadWorkerPool`` to the native-client interface.

    Args:
        pool: A configured ``_ThreadWorkerPool`` (owned by this client; closed
            in ``close()``).
    """

    def __init__(self, pool: _ThreadWorkerPool) -> None:
        self._pool = pool
        self._efd = create_event_notifier()
        self._lock = threading.Lock()
        self._next_future_id = 0
        self._pending: dict[int, _PendingBatch] = {}
        self._completions: deque[
            tuple[int, bool, str, Optional[list[bool]]]
        ] = deque()
        self._closed = False

    # ------------------------------------------------------------------
    # Native-client interface
    # ------------------------------------------------------------------

    def event_fd(self) -> int:
        """Return the poll-able completion fd (signaled once per finished batch)."""
        return self._efd.fileno()

    def submit_batch_set(
        self,
        keys: list[str],
        memviews: list,  # list[memoryview]
    ) -> int:
        """Store a batch of key/value pairs. Returns the batch future id."""
        return self._submit(
            _OP_SET,
            keys,
            lambda i: self._pool.submit_set(keys[i], memviews[i]),
        )

    def submit_batch_get(
        self,
        keys: list[str],
        memviews: list,  # list[memoryview]
    ) -> int:
        """Load a batch of keys into the provided buffers. Returns the batch id."""
        return self._submit(
            _OP_GET,
            keys,
            lambda i: self._pool.submit_get_into(keys[i], memviews[i]),
        )

    def submit_batch_exists(self, keys: list[str]) -> int:
        """Check existence for a batch of keys. Returns the batch future id."""
        return self._submit(
            _OP_EXISTS,
            keys,
            lambda i: self._pool.submit_exists(keys[i]),
        )

    def submit_batch_delete(self, keys: list[str]) -> int:
        """Delete a batch of keys. Returns the batch future id."""
        return self._submit(
            _OP_DELETE,
            keys,
            lambda i: self._pool.submit_delete(keys[i]),
        )

    def drain_completions(
        self,
    ) -> list[tuple[int, bool, str, Optional[list[bool]]]]:
        """Return and clear all finished batches, and consume the eventfd.

        The eventfd is consumed *before* the queue is drained so that a batch
        completing in the small window between consume and drain is never
        lost — at worst it triggers one extra (empty) wakeup on the next poll.
        """
        self._efd.consume()
        with self._lock:
            if not self._completions:
                return []
            drained = list(self._completions)
            self._completions.clear()
        return drained

    def close(self) -> None:
        """Close the underlying pool and the eventfd. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.close()
        self._efd.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _submit(self, op_type: str, keys: list[str], launch) -> int:
        """Register a pending batch and fan out per-key futures.

        The pending entry is created under the lock *before* any future is
        submitted, so a fast callback (which may run inline in this thread
        when a future is already done) always finds its batch. Futures are
        launched outside the lock to avoid re-entrant lock acquisition when a
        callback fires inline.
        """
        n = len(keys)
        with self._lock:
            future_id = self._next_future_id
            self._next_future_id += 1
            # Empty batch: complete immediately so the caller still gets a
            # completion + wakeup and never waits forever.
            if n == 0:
                self._completions.append((future_id, True, "", []))
                notify = True
            else:
                self._pending[future_id] = _PendingBatch(op_type, n)
                notify = False
        if notify:
            self._efd.notify()
            return future_id

        for i in range(n):
            fut = launch(i)
            fut.add_done_callback(self._make_callback(future_id, i))
        return future_id

    def _make_callback(self, future_id: int, index: int):
        def _cb(fut: Future) -> None:
            self._on_key_done(future_id, index, fut)

        return _cb

    def _on_key_done(self, future_id: int, index: int, fut: Future) -> None:
        """Record one key's result; enqueue + signal when the batch finishes."""
        notify = False
        with self._lock:
            pb = self._pending.get(future_id)
            if pb is None:
                # Batch already finalized (should not happen) — ignore.
                return
            try:
                res = fut.result()
                if pb.op_type == _OP_SET:
                    # set returns None; success == no exception.
                    pb.results[index] = True
                else:
                    # get / exists / delete return a bool.
                    pb.results[index] = bool(res)
            except Exception as exc:  # noqa: BLE001 - surface at task level
                pb.results[index] = False
                # Only stores treat a per-key error as a task-level failure;
                # for get/exists/delete a failed key is just a False result.
                if pb.op_type == _OP_SET:
                    pb.ok = False
                if not pb.err:
                    pb.err = str(exc)
                logger.debug(
                    "GlideBatchClient %s key failed (fid=%d idx=%d): %s",
                    pb.op_type,
                    future_id,
                    index,
                    exc,
                )
            pb.remaining -= 1
            if pb.remaining == 0:
                del self._pending[future_id]
                self._completions.append(
                    (future_id, pb.ok, pb.err, pb.results)
                )
                notify = True
        if notify:
            self._efd.notify()
