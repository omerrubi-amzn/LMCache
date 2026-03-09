# SPDX-License-Identifier: Apache-2.0
"""
SyncValkeyConnector — high-throughput Valkey connector using the GLIDE sync
client with a persistent multiprocessing worker pool and shared memory.

Design choices:
- N worker processes, each with its own ``glide_sync.GlideClient`` / Tokio
  runtime, avoiding GIL contention and Tokio ``block_on`` deadlocks.
- ``multiprocessing.SharedMemory`` arena for data transfer between the main
  process and workers without per-operation ``shm_open``/``shm_unlink``.
- Single-key storage (like RESPConnector) to halve Valkey round-trips
  compared to the 2-key metadata/kv_bytes split in the async ValkeyConnector.

Requires ``valkey-glide`` with PRs #5492 (zero-copy SET) and #5493 (buffer GET).
"""

# Standard
from enum import IntEnum, auto
from multiprocessing import shared_memory
from typing import List, Optional, no_type_check
import asyncio
import multiprocessing as mp
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    """Priority levels for the async priority-queue executor."""

    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


def _worker_main(
    host: str,
    port: int,
    username: str,
    password: str,
    database_id: Optional[int],
    request_timeout: int,
    tls_enable: bool,
    req_queue: mp.Queue,
    resp_queue: mp.Queue,
) -> None:
    """Main loop for a persistent worker process.

    Each worker owns an independent ``glide_sync.GlideClient`` (and thus its
    own Tokio runtime), so there is no GIL contention between workers.

    Protocol — request tuples on ``req_queue``:
        ("SET", key_str, shm_name, offset, length)
        ("GET", key_str, shm_name, offset, length)
        ("EXISTS", key_str)
        ("STOP",)

    Responses on ``resp_queue``:
        ("OK",)               — for SET, STOP
        ("OK", bool)          — for EXISTS
        ("OK", bool)          — for GET (True if key found, False if missing)
        ("OK", list[bool])    — for BATCH_EXISTS
        ("ERR", error_string)
    """
    # Imported here because glide_sync is only available in worker processes
    # (the parent process may not have it installed or configured).
    # Third Party
    import glide_sync  # type: ignore[import-untyped]

    credentials = None
    if username or password:
        credentials = glide_sync.ServerCredentials(username, password)

    config_kwargs: dict = {
        "addresses": [glide_sync.NodeAddress(host, port)],
        "request_timeout": request_timeout,
        "use_tls": tls_enable,
    }
    if credentials is not None:
        config_kwargs["credentials"] = credentials
    if database_id is not None:
        config_kwargs["database_id"] = database_id

    config = glide_sync.GlideClientConfiguration(**config_kwargs)
    client = glide_sync.GlideClient.create(config)

    has_buffer_get = "buffer" in client.get.__code__.co_varnames

    while True:
        try:
            msg = req_queue.get()
        except Exception:
            break

        if msg[0] == "STOP":
            resp_queue.put(("OK",))
            break

        try:
            op = msg[0]
            if op == "SET":
                _, key_str, shm_name, offset, length = msg
                shm = shared_memory.SharedMemory(name=shm_name)
                buf = memoryview(shm.buf)[offset : offset + length]
                client.set(key_str.encode(), buf)
                del buf
                shm.close()
                resp_queue.put(("OK",))

            elif op == "GET":
                _, key_str, shm_name, offset, length = msg
                shm = shared_memory.SharedMemory(name=shm_name)
                buf = memoryview(shm.buf)[offset : offset + length]
                if has_buffer_get:
                    result = client.get(key_str.encode(), buffer=buf)
                    found = result is not None
                else:
                    data = client.get(key_str.encode())
                    found = data is not None
                    if data is not None:
                        buf[: len(data)] = data
                del buf
                shm.close()
                resp_queue.put(("OK", found))

            elif op == "EXISTS":
                _, key_str = msg
                exists_result = client.exists([key_str.encode()])
                resp_queue.put(("OK", bool(exists_result)))

            elif op == "BATCH_EXISTS":
                _, key_strs = msg
                results = [bool(client.exists([k.encode()])) for k in key_strs]
                resp_queue.put(("OK", results))

            else:
                resp_queue.put(("ERR", f"unknown op: {op}"))

        except Exception as e:
            resp_queue.put(("ERR", str(e)))

    try:
        client.close()
    except Exception:
        pass


class _SharedArena:
    """Persistent shared memory arena reused across operations.

    Pre-allocates a single ``SharedMemory`` segment sized for
    ``num_slots × slot_size`` bytes. Slots are acquired/released via a
    simple free-list so no ``shm_open``/``shm_unlink`` syscalls occur on
    the hot path.

    Args:
        num_slots: Number of slots (typically ``num_workers``).
        slot_size: Size of each slot in bytes (typically ``full_chunk_size_bytes``).
    """

    def __init__(self, num_slots: int, slot_size: int):
        self.num_slots = num_slots
        self.slot_size = slot_size
        total = num_slots * slot_size
        self._shm = shared_memory.SharedMemory(create=True, size=max(total, 1))
        self._buf = memoryview(self._shm.buf)
        self._lock = threading.Lock()
        self._free: list[int] = list(range(num_slots))
        logger.info(
            "SharedArena: %d slots × %d bytes = %d bytes, shm=%s",
            num_slots,
            slot_size,
            total,
            self._shm.name,
        )

    @property
    def name(self) -> str:
        """The shared memory segment name (for passing to workers).

        Returns:
            The OS-level name of the shared memory segment.
        """
        return self._shm.name

    def acquire(self) -> int:
        """Acquire a free slot index.

        Returns:
            The slot index.

        Raises:
            RuntimeError: If no slots are available.
        """
        with self._lock:
            if not self._free:
                raise RuntimeError("SharedArena: no free slots")
            return self._free.pop()

    def release(self, slot: int) -> None:
        """Return a slot to the free list.

        Args:
            slot: The slot index to release.
        """
        with self._lock:
            self._free.append(slot)

    def offset(self, slot: int) -> int:
        """Byte offset for a given slot.

        Args:
            slot: The slot index.

        Returns:
            The byte offset into the shared memory segment.
        """
        return slot * self.slot_size

    def slot_view(self, slot: int) -> memoryview:
        """Get a memoryview for a specific slot.

        Args:
            slot: The slot index.

        Returns:
            A memoryview covering the slot's bytes.
        """
        off = slot * self.slot_size
        return self._buf[off : off + self.slot_size]

    def close(self) -> None:
        """Release and unlink the shared memory segment."""
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


class _WorkerPool:
    """Manages a pool of persistent worker processes for GLIDE sync I/O.

    Each worker has its own ``glide_sync.GlideClient`` and Tokio runtime.
    Communication uses ``multiprocessing.Queue`` for control messages and
    ``SharedMemory`` for KV data (zero-copy across processes).

    Args:
        host: Valkey server hostname.
        port: Valkey server port.
        num_workers: Number of worker processes.
        username: Valkey authentication username.
        password: Valkey authentication password.
        database_id: Optional Valkey database ID.
        request_timeout: GLIDE client request timeout in milliseconds.
        tls_enable: Whether to use TLS for Valkey connections.
    """

    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int,
        username: str,
        password: str,
        database_id: Optional[int],
        request_timeout: int = 120_000,
        tls_enable: bool = False,
    ):
        self.num_workers = num_workers
        self._req_queues: list[mp.Queue] = []
        self._resp_queues: list[mp.Queue] = []
        self._processes: list[mp.Process] = []
        self._closed = False

        for _ in range(num_workers):
            req_q: mp.Queue = mp.Queue()
            resp_q: mp.Queue = mp.Queue()
            p = mp.Process(
                target=_worker_main,
                args=(
                    host,
                    port,
                    username,
                    password,
                    database_id,
                    request_timeout,
                    tls_enable,
                    req_q,
                    resp_q,
                ),
                daemon=True,
            )
            p.start()
            self._req_queues.append(req_q)
            self._resp_queues.append(resp_q)
            self._processes.append(p)

        logger.info("SyncValkey worker pool: %d processes started", num_workers)

    def submit(self, worker_id: int, msg: tuple) -> None:
        """Send a request to a specific worker.

        Args:
            worker_id: Index of the target worker process.
            msg: Request tuple (see ``_worker_main`` docstring).
        """
        self._req_queues[worker_id].put(msg)

    def collect(self, worker_id: int, timeout: float = 60.0) -> tuple:
        """Collect a response from a specific worker.

        Args:
            worker_id: Index of the worker to collect from.
            timeout: Maximum seconds to wait.

        Returns:
            Response tuple from the worker.

        Raises:
            RuntimeError: If the worker returns an error or times out.
        """
        resp = self._resp_queues[worker_id].get(timeout=timeout)
        if resp[0] == "ERR":
            raise RuntimeError(f"SyncValkey worker error: {resp[1]}")
        return resp

    def close(self) -> None:
        """Shut down all worker processes gracefully."""
        if self._closed:
            return
        self._closed = True
        for i in range(self.num_workers):
            try:
                self._req_queues[i].put(("STOP",))
            except Exception:
                pass
        for i, p in enumerate(self._processes):
            try:
                self._resp_queues[i].get(timeout=5.0)
            except Exception:
                pass
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        logger.info("SyncValkey worker pool closed")


class SyncValkeyConnector(RemoteConnector):
    """High-throughput Valkey connector using GLIDE sync client with
    multiprocessing.

    Uses N persistent worker processes (each with its own
    ``glide_sync.GlideClient``) and ``SharedMemory`` for data transfer.
    Single-key storage (like RESPConnector) halves Valkey round-trips
    compared to the 2-key metadata/kv_bytes split in the async
    ValkeyConnector.

    Args:
        host: Valkey server hostname.
        port: Valkey server port.
        loop: Asyncio event loop for the priority-queue executor.
        local_cpu_backend: Backend for allocating CPU memory objects.
        num_workers: Number of worker processes (default 8).
        username: Valkey authentication username.
        password: Valkey authentication password.
        database_id: Optional Valkey database ID.
        tls_enable: Whether to use TLS for Valkey connections.
    """

    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        num_workers: int = 8,
        username: str = "",
        password: str = "",
        database_id: Optional[int] = None,
        tls_enable: bool = False,
    ):
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self._pool = _WorkerPool(
            host,
            port,
            num_workers,
            username,
            password,
            database_id,
            tls_enable=tls_enable,
        )
        self._pq_executor = AsyncPQExecutor(loop)
        # 2× slots enables pipelined batching: workers process the first
        # wave while the main thread fills the second wave of slots.
        self._arena = _SharedArena(2 * num_workers, self.full_chunk_size_bytes)
        self._rr_counter = 0  # round-robin counter for single-key ops

    def _next_worker(self) -> int:
        """Return the next worker index using round-robin scheduling.

        Returns:
            Worker index in ``[0, num_workers)``.
        """
        wid = self._rr_counter % self.num_workers
        self._rr_counter += 1
        return wid

    async def _exists(self, key: CacheEngineKey) -> bool:
        key_str = key.to_string()
        wid = self._next_worker()
        self._pool.submit(wid, ("EXISTS", key_str))
        resp = self._pool.collect(wid)
        return resp[1]

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if a key exists in Valkey.

        Args:
            key: The cache engine key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return await self._pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronously check if a key exists in Valkey.

        Args:
            key: The cache engine key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        key_str = key.to_string()
        wid = self._next_worker()
        self._pool.submit(wid, ("EXISTS", key_str))
        resp = self._pool.collect(wid)
        return resp[1]

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes, self.meta_dtypes, self.meta_fmt
        )

        byte_array = memory_obj.byte_array
        if not isinstance(byte_array, memoryview):
            byte_array = memoryview(byte_array)
        dst = byte_array.cast("B") if byte_array.format != "B" else byte_array
        size = len(dst)

        wid = self._next_worker()
        slot = self._arena.acquire()
        try:
            off = self._arena.offset(slot)
            self._pool.submit(wid, ("GET", key_str, self._arena.name, off, size))
            resp = self._pool.collect(wid)
            found = resp[1] if len(resp) > 1 else True
            if not found:
                memory_obj.ref_count_down()
                return None
            dst[:size] = self._arena.slot_view(slot)[:size]
        except Exception:
            memory_obj.ref_count_down()
            raise
        finally:
            self._arena.release(slot)

        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve a memory object from Valkey by key.

        Args:
            key: The cache engine key.

        Returns:
            The retrieved MemoryObj, or None if the key does not exist.
        """
        return await self._pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        key_str = key.to_string()
        byte_array = memory_obj.byte_array
        if not isinstance(byte_array, memoryview):
            byte_array = memoryview(byte_array)
        src = byte_array.cast("B") if byte_array.format != "B" else byte_array
        size = len(src)

        wid = self._next_worker()
        slot = self._arena.acquire()
        try:
            self._arena.slot_view(slot)[:size] = src[:size]
            off = self._arena.offset(slot)
            self._pool.submit(wid, ("SET", key_str, self._arena.name, off, size))
            self._pool.collect(wid)
        finally:
            self._arena.release(slot)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Store a memory object in Valkey.

        Args:
            key: The cache engine key.
            memory_obj: The memory object to store.
        """
        await self._pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    def support_batched_put(self) -> bool:
        """Returns True — batched put is supported.

        Returns:
            True
        """
        return True

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> None:
        n = len(keys)
        key_strs = [k.to_string() for k in keys]

        # Prepare source buffers
        src_bufs = []
        for mobj in memory_objs:
            ba = mobj.byte_array
            if not isinstance(ba, memoryview):
                ba = memoryview(ba)
            if ba.format != "B":
                ba = ba.cast("B")
            src_bufs.append(ba)

        chunk_size = len(src_bufs[0])

        # Pipelined round-robin: submit all items without wave barriers.
        # With 2×num_workers arena slots we can keep up to 2 items
        # in-flight per worker, overlapping I/O with memcpy.
        #
        # in_flight tracks (worker_id, slot) in submission order so we
        # can collect responses FIFO when we run out of arena slots.
        in_flight: list[tuple[int, int]] = []
        max_in_flight = self._arena.num_slots

        for i in range(n):
            # If all slots are busy, drain the oldest in-flight item
            if len(in_flight) >= max_in_flight:
                wid_old, slot_old = in_flight.pop(0)
                self._pool.collect(wid_old)
                self._arena.release(slot_old)

            wid = i % self.num_workers
            slot = self._arena.acquire()
            self._arena.slot_view(slot)[:chunk_size] = src_bufs[i][:chunk_size]
            self._pool.submit(
                wid,
                (
                    "SET",
                    key_strs[i],
                    self._arena.name,
                    self._arena.offset(slot),
                    chunk_size,
                ),
            )
            in_flight.append((wid, slot))

        # Drain remaining in-flight items
        for wid, slot in in_flight:
            self._pool.collect(wid)
            self._arena.release(slot)

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ) -> None:
        """Store multiple memory objects in Valkey in parallel.

        Keys are distributed across worker processes round-robin.

        Args:
            keys: List of cache engine keys.
            memory_objs: List of memory objects to store.
        """
        await self._pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    def support_batched_get(self) -> bool:
        """Returns True — batched get is supported.

        Returns:
            True
        """
        return True

    async def _batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        n = len(keys)
        key_strs = [k.to_string() for k in keys]

        memory_objs = [
            self.local_cpu_backend.allocate(
                self.meta_shapes, self.meta_dtypes, self.meta_fmt
            )
            for _ in keys
        ]

        chunk_size = memory_objs[0].get_size()

        try:
            # Pipelined round-robin: submit all items without wave
            # barriers.  in_flight tracks (worker_id, slot, item_idx)
            # in submission order; we drain the oldest when out of slots.
            in_flight: list[tuple[int, int, int]] = []
            max_in_flight = self._arena.num_slots

            for i in range(n):
                if len(in_flight) >= max_in_flight:
                    wid_old, slot_old, idx_old = in_flight.pop(0)
                    self._pool.collect(wid_old)
                    dst = memory_objs[idx_old].byte_array
                    if not isinstance(dst, memoryview):
                        dst = memoryview(dst)
                    if dst.format != "B":
                        dst = dst.cast("B")
                    dst[:chunk_size] = self._arena.slot_view(slot_old)[:chunk_size]
                    self._arena.release(slot_old)

                wid = i % self.num_workers
                slot = self._arena.acquire()
                self._pool.submit(
                    wid,
                    (
                        "GET",
                        key_strs[i],
                        self._arena.name,
                        self._arena.offset(slot),
                        chunk_size,
                    ),
                )
                in_flight.append((wid, slot, i))

            # Drain remaining in-flight items
            for wid, slot, idx in in_flight:
                self._pool.collect(wid)
                dst = memory_objs[idx].byte_array
                if not isinstance(dst, memoryview):
                    dst = memoryview(dst)
                if dst.format != "B":
                    dst = dst.cast("B")
                dst[:chunk_size] = self._arena.slot_view(slot)[:chunk_size]
                self._arena.release(slot)

        except Exception:
            for mobj in memory_objs:
                mobj.ref_count_down()
            raise

        return memory_objs

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Retrieve multiple memory objects from Valkey in parallel.

        Args:
            keys: List of cache engine keys.

        Returns:
            List of retrieved MemoryObj instances.
        """
        return await self._pq_executor.submit_job(
            self._batched_get, keys=keys, priority=Priorities.GET
        )

    def support_batched_contains(self) -> bool:
        """Returns True — synchronous batched contains is supported.

        Returns:
            True
        """
        return True

    def _count_consecutive_exists(self, keys: List[CacheEngineKey]) -> int:
        """Check how many consecutive keys exist (prefix match) via the pool.

        Args:
            keys: List of keys to check.

        Returns:
            Number of consecutive keys that exist from the start.
        """
        key_strs = [k.to_string() for k in keys]
        wid = self._next_worker()
        self._pool.submit(wid, ("BATCH_EXISTS", key_strs))
        resp = self._pool.collect(wid)
        results = resp[1]
        count = 0
        for r in results:
            if not r:
                return count
            count += 1
        return count

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        """Synchronously check how many consecutive keys exist (prefix match).

        Args:
            keys: List of keys to check.

        Returns:
            Number of consecutive keys that exist from the start.
        """
        return self._count_consecutive_exists(keys)

    def support_batched_async_contains(self) -> bool:
        """Returns True — async batched contains is supported.

        Returns:
            True
        """
        return True

    async def _batched_async_contains(self, keys: List[CacheEngineKey]) -> int:
        return self._count_consecutive_exists(keys)

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Asynchronously check how many consecutive keys exist (prefix match).

        Args:
            lookup_id: Identifier for this lookup operation.
            keys: List of keys to check.
            pin: Whether to pin the keys (unused).

        Returns:
            Number of consecutive keys that exist from the start.
        """
        return await self._pq_executor.submit_job(
            self._batched_async_contains, keys=keys, priority=Priorities.PREFETCH
        )

    def support_batched_get_non_blocking(self) -> bool:
        """Returns True — non-blocking batched get is supported.

        Returns:
            True
        """
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """Non-blocking batched get (delegates to batched_get via executor).

        Args:
            lookup_id: Identifier for this lookup operation.
            keys: List of keys to get.

        Returns:
            List of retrieved MemoryObj instances.
        """
        return await self._pq_executor.submit_job(
            self._batched_get, keys=keys, priority=Priorities.PREFETCH
        )

    @no_type_check
    async def list(self) -> List[str]:
        """List all keys (not implemented).

        Returns:
            Empty list.
        """
        return []

    async def close(self) -> None:
        """Shut down the executor, worker pool, and shared memory arena."""
        self._pq_executor.shutdown(wait=True)
        self._pool.close()
        self._arena.close()
        logger.info("Closed SyncValkeyConnector")
