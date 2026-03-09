# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for SyncValkeyConnector.

These tests verify the sync Valkey connector implementation, including:
- Basic operations (exists, get, set)
- Batch operations (batch_get, batch_put, batch_exists)
- Error handling
- Worker scaling

The worker pool is mocked to avoid requiring glide_sync or a real Valkey server.
"""

# Standard
from multiprocessing import shared_memory
from unittest.mock import patch
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import PinMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import LocalCPUBackend
from lmcache.v1.storage_backend.connector import CreateConnector

# Local
from ...conftest import MockSyncGlideClient
from ..utils import (
    check_mem_obj_equal,
    close_asyncio_loop,
    dumb_cache_engine_key,
    init_asyncio_loop,
)


class MockWorkerPool:
    """In-memory mock of _WorkerPool that uses MockSyncGlideClient directly.

    Runs all operations in-process (no multiprocessing) so tests don't need
    glide_sync or a real Valkey server.
    """

    def __init__(self, *args, **kwargs):
        self.num_workers = kwargs.get("num_workers", 8)
        if isinstance(args[2] if len(args) > 2 else self.num_workers, int):
            self.num_workers = args[2] if len(args) > 2 else self.num_workers
        self._client = MockSyncGlideClient()
        self._closed = False
        self._pending: dict[int, list[tuple]] = {}

    def submit(self, worker_id: int, msg: tuple) -> None:
        """Queue request for the given worker (synchronous mock)."""
        self._pending.setdefault(worker_id, []).append(msg)

    def collect(self, worker_id: int, timeout: float = 60.0) -> tuple:
        """Execute the next pending request for worker_id."""
        msg = self._pending[worker_id].pop(0)
        op = msg[0]

        if op == "SET":
            _, key_str, shm_name, offset, length = msg
            shm = shared_memory.SharedMemory(name=shm_name)
            buf = memoryview(shm.buf)[offset : offset + length]
            self._client.set(key_str.encode(), buf)
            del buf
            shm.close()
            return ("OK",)

        elif op == "GET":
            _, key_str, shm_name, offset, length = msg
            shm = shared_memory.SharedMemory(name=shm_name)
            buf = memoryview(shm.buf)[offset : offset + length]
            data = self._client.get(key_str.encode())
            found = data is not None
            if found:
                buf[: len(data)] = data
            del buf
            shm.close()
            return ("OK", found)

        elif op == "EXISTS":
            _, key_str = msg
            result = self._client.exists([key_str.encode()])
            return ("OK", bool(result))

        elif op == "BATCH_EXISTS":
            _, key_strs = msg
            results = [bool(self._client.exists([k.encode()])) for k in key_strs]
            return ("OK", results)

        return ("ERR", f"unknown op: {op}")

    def close(self) -> None:
        self._closed = True


def _get_metadata():
    """Helper to create test metadata."""
    kv_shape = (32, 2, 256, 8, 128)
    return LMCacheMetadata(
        model_name="test-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        use_mla=False,
    )


def _create_local_cpu_backend(memory_allocator, config=None):
    """Helper to create a local CPU backend for testing."""
    if config is None:
        config = LMCacheEngineConfig.from_defaults(
            extra_config={"valkey_sync_num_workers": 4}
        )
    metadata = _get_metadata()
    return LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )


@pytest.fixture(autouse=True)
def mock_worker_pool():
    """Replace _WorkerPool with in-memory mock so tests never spawn processes."""
    MockSyncGlideClient.reset_store()
    with patch(
        "lmcache.v1.storage_backend.connector.sync_valkey_connector._WorkerPool",
        MockWorkerPool,
    ):
        yield


@pytest.fixture
def sync_valkey_url():
    """URL for testing."""
    return "valkey-sync://mock.local:0"


@pytest.fixture
def sync_valkey_config():
    """Config for SyncValkeyConnector testing."""
    return LMCacheEngineConfig.from_defaults(
        extra_config={"valkey_sync_num_workers": 4}
    )


@pytest.fixture
def local_backend():
    """Create a local CPU backend for testing."""
    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    backend = _create_local_cpu_backend(memory_allocator)
    yield backend
    backend.close()


def test_sync_valkey_basic_operations(
    sync_valkey_url, local_backend, sync_valkey_config, autorelease_v1
):
    """Test basic operations: exists, put, get."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(
                sync_valkey_url, async_loop, local_backend, sync_valkey_config
            )
        )

        random_key = dumb_cache_engine_key()

        # Key doesn't exist initially
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(random_key), async_loop
        )
        assert not future.result(), "Key should not exist initially"

        # Create and store test data
        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16
        memory_obj = local_backend.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        torch.manual_seed(42)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        # Put data
        future = asyncio.run_coroutine_threadsafe(
            connector.put(random_key, memory_obj), async_loop
        )
        future.result()

        # Key exists after put
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(random_key), async_loop
        )
        assert future.result(), "Key should exist after put"

        # Get and verify data
        future = asyncio.run_coroutine_threadsafe(connector.get(random_key), async_loop)
        retrieved = future.result()
        check_mem_obj_equal([retrieved], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_sync_valkey_batch_operations(
    sync_valkey_url, local_backend, sync_valkey_config, autorelease_v1
):
    """Test batch operations: batched_put, batched_get, batched_async_contains."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(
                sync_valkey_url, async_loop, local_backend, sync_valkey_config
            )
        )

        num_keys = 10
        keys = [dumb_cache_engine_key(i) for i in range(num_keys)]

        # Batch exists — all should be False initially
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_async_contains("test_lookup", keys), async_loop
        )
        assert future.result() == 0, "No keys should exist initially"

        # Create memory objects
        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16
        memory_objs = []

        for i in range(num_keys):
            memory_obj = local_backend.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()
            torch.manual_seed(42 + i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))
            memory_objs.append(memory_obj)

        # Batch put
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys, memory_objs), async_loop
        )
        future.result()

        # Batch exists — all should be True now
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_async_contains("test_lookup", keys), async_loop
        )
        assert future.result() == num_keys, "All keys should exist after batch_put"

        # Batch get and verify
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get(keys), async_loop
        )
        retrieved_objs = future.result()

        assert len(retrieved_objs) == num_keys
        check_mem_obj_equal(retrieved_objs, memory_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_sync_valkey_nonexistent_key(
    sync_valkey_url, local_backend, sync_valkey_config, autorelease_v1
):
    """Test exists and get on a non-existent key."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(
                sync_valkey_url, async_loop, local_backend, sync_valkey_config
            )
        )

        nonexistent_key = dumb_cache_engine_key()

        # exists should return False
        future = asyncio.run_coroutine_threadsafe(
            connector.exists(nonexistent_key), async_loop
        )
        assert not future.result()

        # get should return None for missing keys
        future = asyncio.run_coroutine_threadsafe(
            connector.get(nonexistent_key), async_loop
        )
        assert future.result() is None, "get() should return None for missing key"

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_sync_valkey_sequential_operations(
    sync_valkey_url, local_backend, sync_valkey_config, autorelease_v1
):
    """Test multiple sequential put/get cycles."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(
                sync_valkey_url, async_loop, local_backend, sync_valkey_config
            )
        )

        for i in range(5):
            key = dumb_cache_engine_key(i)
            num_tokens = 256
            mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
            dtype = torch.bfloat16
            memory_obj = local_backend.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()

            torch.manual_seed(1000 + i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            future.result()

            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            retrieved = future.result()
            check_mem_obj_equal([retrieved], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_sync_valkey_concurrent_operations(
    sync_valkey_url, local_backend, sync_valkey_config, autorelease_v1
):
    """Test concurrent put/get operations."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(
                sync_valkey_url, async_loop, local_backend, sync_valkey_config
            )
        )

        num_concurrent = 5
        keys = [dumb_cache_engine_key(i) for i in range(num_concurrent)]
        memory_objs = []

        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16

        put_futures = []
        for i, key in enumerate(keys):
            memory_obj = local_backend.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()
            torch.manual_seed(2000 + i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))
            memory_objs.append(memory_obj)

            future = asyncio.run_coroutine_threadsafe(
                connector.put(key, memory_obj), async_loop
            )
            put_futures.append(future)

        for future in put_futures:
            future.result()

        get_futures = []
        for key in keys:
            future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
            get_futures.append(future)

        retrieved_objs = [f.result() for f in get_futures]
        check_mem_obj_equal(retrieved_objs, memory_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_sync_valkey_exists_sync(
    sync_valkey_url, local_backend, sync_valkey_config, autorelease_v1
):
    """Test synchronous exists method."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(
                sync_valkey_url, async_loop, local_backend, sync_valkey_config
            )
        )

        key = dumb_cache_engine_key()
        assert not connector.exists_sync(key)

        # Put data
        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16
        memory_obj = local_backend.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result()

        assert connector.exists_sync(key)

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_sync_valkey_batched_contains_prefix(
    sync_valkey_url, local_backend, sync_valkey_config, autorelease_v1
):
    """Test that batched_contains returns prefix count correctly."""
    async_loop, async_thread = init_asyncio_loop()

    try:
        connector = autorelease_v1(
            CreateConnector(
                sync_valkey_url, async_loop, local_backend, sync_valkey_config
            )
        )

        keys = [dumb_cache_engine_key(i) for i in range(5)]
        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16

        # Put only first 3 keys
        for i in range(3):
            memory_obj = local_backend.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()
            future = asyncio.run_coroutine_threadsafe(
                connector.put(keys[i], memory_obj), async_loop
            )
            future.result()

        # batched_contains should return 3 (prefix match stops at key[3])
        count = connector.batched_contains(keys)
        assert count == 3, f"Expected 3 consecutive keys, got {count}"

    finally:
        close_asyncio_loop(async_loop, async_thread)


def test_sync_valkey_different_chunk_sizes(autorelease_v1):
    """Test that the connector works with different chunk sizes."""
    async_loop, async_thread = init_asyncio_loop()

    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    config = LMCacheEngineConfig.from_defaults(
        extra_config={"valkey_sync_num_workers": 4}
    )

    kv_shape = (32, 2, 512, 8, 128)
    dtype = torch.bfloat16
    metadata = LMCacheMetadata(
        model_name="test-model-large",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=kv_shape,
        use_mla=False,
        chunk_size=512,
    )
    local_backend = LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )

    try:
        connector = autorelease_v1(
            CreateConnector(
                "valkey-sync://mock.local:0",
                async_loop,
                local_backend,
                config,
            )
        )

        key = dumb_cache_engine_key()
        mem_obj_shape = torch.Size([2, 32, 512, 1024])
        memory_obj = local_backend.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        torch.manual_seed(100)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result()

        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved = future.result()
        check_mem_obj_equal([retrieved], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)
        local_backend.close()


def test_sync_valkey_pipelined_batch_exceeds_arena(
    sync_valkey_url, sync_valkey_config, autorelease_v1
):
    """Test batched put/get when batch size > arena slots (2×num_workers).

    With num_workers=4 the arena has 8 slots.  A batch of 12 keys forces
    the sliding-window pipeline to drain old items mid-batch, verifying
    that the FIFO drain logic works correctly and no data is lost.

    Uses a dedicated 2 GB allocator so the 12 put objects + 12 get objects
    (~768 MB total at 32 MB each) fit without blocking.
    """
    async_loop, async_thread = init_asyncio_loop()

    memory_allocator = PinMemoryAllocator(2 * 1024 * 1024 * 1024)
    local_backend = _create_local_cpu_backend(memory_allocator, sync_valkey_config)

    try:
        connector = autorelease_v1(
            CreateConnector(
                sync_valkey_url, async_loop, local_backend, sync_valkey_config
            )
        )

        num_keys = 12  # > 2 × 4 workers = 8 arena slots
        keys = [dumb_cache_engine_key(i) for i in range(num_keys)]

        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16
        memory_objs = []

        for i in range(num_keys):
            memory_obj = local_backend.allocate(mem_obj_shape, dtype)
            memory_obj.ref_count_up()
            torch.manual_seed(5000 + i)
            test_tensor = torch.randint(
                0, 100, memory_obj.raw_data.shape, dtype=torch.int64
            )
            memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))
            memory_objs.append(memory_obj)

        # Batch put all 12
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_put(keys, memory_objs), async_loop
        )
        future.result()

        # All 12 should exist
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_async_contains("test_lookup", keys), async_loop
        )
        assert future.result() == num_keys

        # Batch get and verify every item matches
        future = asyncio.run_coroutine_threadsafe(
            connector.batched_get(keys), async_loop
        )
        retrieved_objs = future.result()

        assert len(retrieved_objs) == num_keys
        check_mem_obj_equal(retrieved_objs, memory_objs)

    finally:
        close_asyncio_loop(async_loop, async_thread)
        local_backend.close()


@pytest.mark.parametrize("num_workers", [1, 4, 8])
def test_sync_valkey_worker_scaling(num_workers, autorelease_v1):
    """Test SyncValkeyConnector with different numbers of worker processes."""
    async_loop, async_thread = init_asyncio_loop()

    memory_allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    config = LMCacheEngineConfig.from_defaults(
        extra_config={"valkey_sync_num_workers": num_workers}
    )
    metadata = _get_metadata()
    local_backend = LocalCPUBackend(
        config=config, metadata=metadata, memory_allocator=memory_allocator
    )

    try:
        connector = autorelease_v1(
            CreateConnector(
                "valkey-sync://mock.local:0",
                async_loop,
                local_backend,
                config,
            )
        )

        key = dumb_cache_engine_key()
        num_tokens = 256
        mem_obj_shape = torch.Size([2, 32, num_tokens, 1024])
        dtype = torch.bfloat16
        memory_obj = local_backend.allocate(mem_obj_shape, dtype)
        memory_obj.ref_count_up()

        torch.manual_seed(3000)
        test_tensor = torch.randint(
            0, 100, memory_obj.raw_data.shape, dtype=torch.int64
        )
        memory_obj.raw_data.copy_(test_tensor.to(torch.float32).to(dtype))

        future = asyncio.run_coroutine_threadsafe(
            connector.put(key, memory_obj), async_loop
        )
        future.result()

        future = asyncio.run_coroutine_threadsafe(connector.get(key), async_loop)
        retrieved = future.result()
        check_mem_obj_equal([retrieved], [memory_obj])

    finally:
        close_asyncio_loop(async_loop, async_thread)
        local_backend.close()
