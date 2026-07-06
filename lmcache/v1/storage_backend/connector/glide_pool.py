# SPDX-License-Identifier: Apache-2.0
"""
Shared GLIDE sync thread-worker pool.

``_ThreadWorkerPool`` manages N worker threads, each with its own GLIDE sync
client (``GlideClient`` standalone or ``GlideClusterClient`` cluster) via
``threading.local()``. Because GLIDE releases the GIL during its FFI calls,
this enables true parallel Valkey I/O from a single process.

This module is the single source of truth for the pool and is shared by:

- ``valkey_connector.py`` — the non-MP ``RemoteConnector`` (``valkey://`` scheme).
- ``distributed/l2_adapters/glide_batch_client.py`` — the MP-mode batch client
  wrapped by ``NativeConnectorL2Adapter``.

Design choices:
- Per-thread clients enable overlapping round-trips (wall time ~= ceil(N_ops /
  num_workers) * RTT instead of N_ops * RTT).
- Zero-copy buffer-GET into a per-thread scratch buffer when the GLIDE client
  supports ``get(key, buffer=...)``, with a graceful fallback to plain GET.
- Single-key, fixed-size storage (like RESP) — no per-chunk metadata.
- Optional per-key TTL so Valkey/Redis ``volatile-*`` eviction policies can
  reclaim L2 keys under memory pressure.

Requires ``valkey-glide`` >= 2.3. Zero-copy is used when available
(GLIDE PRs #5492 / #5493); otherwise the pool falls back transparently.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Optional
import inspect
import threading

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # Third Party
    import glide_sync  # type: ignore[import-untyped]

logger = init_logger(__name__)

#: Default request timeout (seconds).
DEFAULT_REQUEST_TIMEOUT_SECS: float = 5.0
#: Default connection timeout (seconds).
DEFAULT_CONNECTION_TIMEOUT_SECS: float = 10.0
#: Default key TTL (seconds) used when the TTL feature flag is enabled but no
#: explicit ``valkey_ttl_sec`` is configured (24 hours).
DEFAULT_TTL_SECS: int = 86400


class Priorities(IntEnum):
    """Operation priorities for the ``AsyncPQExecutor`` (non-MP path).

    Lower numeric value = higher priority.  Matches the scheme used by
    ``RESPConnector`` so that exists/peek checks run before bulk writes.
    """

    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class _ThreadWorkerPool:
    """Manages a pool of threads, each with its own GLIDE sync client.

    Each thread gets an independent GLIDE sync client
    (``GlideClient`` or ``GlideClusterClient``) via
    ``threading.local()``, enabling true parallel I/O when the GIL is
    released during FFI calls.

    Args:
        host: Valkey server hostname.
        port: Valkey server port.
        num_workers: Number of worker threads.
        username: Valkey authentication username.
        password: Valkey authentication password.
        request_timeout: Timeout in seconds for GLIDE requests and
            Future.result() calls.
        connection_timeout: Timeout in seconds for initial GLIDE client
            connections and thread pool warmup.
        tls_enable: Whether to use TLS for Valkey connections.
        cluster_mode: If True, use GlideClusterClient; else GlideClient.
        database_id: Database ID for standalone mode (ignored in cluster).
        ttl_seconds: If set, every key is written with this expiry (in
            seconds) so Valkey/Redis ``volatile-*`` eviction policies can
            reclaim L2 cache keys under memory pressure. ``None`` (default)
            disables TTL — keys are persisted without expiry.
    """

    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int,
        username: str,
        password: str,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECS,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT_SECS,
        tls_enable: bool = False,
        cluster_mode: bool = False,
        database_id: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        self.num_workers = num_workers
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._request_timeout = request_timeout
        self._request_timeout_ms = int(request_timeout * 1000)
        self._connection_timeout_ms = int(connection_timeout * 1000)
        self._tls_enable = tls_enable
        self._cluster_mode = cluster_mode
        self._database_id = database_id
        self._ttl_seconds = ttl_seconds
        self._expiry_obj = None  # lazily built ExpirySet (see _do_set)
        self._local = threading.local()
        self._has_buffer_get: Optional[bool] = None

        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="valkey",
        )
        # Warm up: create a client on each thread
        futs = [self._executor.submit(self._get_client) for _ in range(num_workers)]
        for f in futs:
            f.result(timeout=connection_timeout)
        mode_str = "cluster" if cluster_mode else "standalone"
        logger.info(
            "Valkey thread pool: %d threads, mode=%s, per-thread clients, "
            "buffer_get=%s, ttl_seconds=%s",
            num_workers,
            mode_str,
            self._has_buffer_get,
            ttl_seconds,
        )

    def _get_client(self):  # type: ignore[no-untyped-def]
        """Get or create the per-thread GLIDE sync client.

        Creates a ``GlideClusterClient`` (cluster mode) or ``GlideClient``
        (standalone mode) depending on the ``cluster_mode`` flag.
        """
        # Third Party
        import glide_sync  # type: ignore[import-untyped]

        client = getattr(self._local, "client", None)
        if client is not None:
            return client

        credentials = None
        if self._username or self._password:
            credentials = glide_sync.ServerCredentials(self._username, self._password)

        address = glide_sync.NodeAddress(self._host, self._port)

        if self._cluster_mode:
            advanced = glide_sync.AdvancedGlideClusterClientConfiguration(
                connection_timeout=self._connection_timeout_ms,
            )
            config_kwargs: dict = {
                "addresses": [address],
                "request_timeout": self._request_timeout_ms,
                "use_tls": self._tls_enable,
                "advanced_config": advanced,
            }
            if credentials is not None:
                config_kwargs["credentials"] = credentials
            config = glide_sync.GlideClusterClientConfiguration(**config_kwargs)
            client = glide_sync.GlideClusterClient.create(config)
        else:
            # Standalone mode — supports database_id and advanced config
            advanced = glide_sync.AdvancedGlideClientConfiguration(
                connection_timeout=self._connection_timeout_ms,
            )
            config_kwargs = {
                "addresses": [address],
                "request_timeout": self._request_timeout_ms,
                "use_tls": self._tls_enable,
                "advanced_config": advanced,
            }
            if credentials is not None:
                config_kwargs["credentials"] = credentials
            if self._database_id is not None:
                config_kwargs["database_id"] = self._database_id
            config = glide_sync.GlideClientConfiguration(**config_kwargs)
            client = glide_sync.GlideClient.create(config)

        self._local.client = client

        if self._has_buffer_get is None:
            self._has_buffer_get = "buffer" in inspect.signature(client.get).parameters

        return client

    @property
    def has_buffer_get(self) -> bool:
        """Whether the GLIDE client supports buffer GET."""
        if self._has_buffer_get is None:
            self._executor.submit(self._get_client).result(
                timeout=self._connection_timeout_ms / 1000
            )
        return bool(self._has_buffer_get)

    def _build_expiry(self) -> "glide_sync.ExpirySet":
        """Lazily build (and cache) the GLIDE ``ExpirySet`` for SET calls.

        Built lazily so ``glide_sync`` is only imported on worker threads
        (matching ``_get_client``). The resulting ``ExpirySet`` is an
        immutable value object, so it is safe to share across threads; a
        benign race during lazy init just rebuilds an equivalent object.
        """
        exp = self._expiry_obj
        if exp is None:
            if self._ttl_seconds is None:
                raise ValueError("ttl_seconds must be set to build an ExpirySet")
            # Third Party
            import glide_sync  # type: ignore[import-untyped]

            exp = glide_sync.ExpirySet(glide_sync.ExpiryType.SEC, self._ttl_seconds)
            self._expiry_obj = exp
        return exp

    def _do_set(self, key_str: str, data: bytes) -> None:
        """SET a key (runs on a worker thread).

        When a TTL is configured (``ttl_seconds`` is not None), the key is
        written with an expiry so that Valkey/Redis ``volatile-*`` eviction
        policies can reclaim it once the node reaches ``maxmemory``. Without
        a TTL the key is persisted indefinitely (legacy behavior).
        """
        expiry = self._build_expiry() if self._ttl_seconds is not None else None
        self._get_client().set(key_str.encode(), data, expiry=expiry)

    def _get_scratch(self, size: int) -> bytearray:
        """Per-thread reusable staging buffer for buffer-GET (issue #6215).

        GLIDE writes into this thread-private bytearray (not slab memory),
        then we copy into the destination under the GIL.  Eliminates the
        dangling-slot race without pinning or touching ref_count/pin_count.
        The buffer is grown on demand and reused across GETs on the same
        worker thread, so there is no per-call allocation.
        """
        buf = getattr(self._local, "scratch", None)
        if buf is None or len(buf) < size:
            buf = bytearray(size)
            self._local.scratch = buf
        return buf

    def _do_get_into(self, key_str: str, buf: memoryview) -> bool:
        """GET a key into a buffer (runs on a worker thread).

        Safety (issue #6215): GLIDE's native buffer-GET writes with the GIL
        released, so it must never write directly into ``buf`` when ``buf``
        aliases slab-allocator memory that another thread could recycle
        mid-write.  We stage into a thread-private scratch buffer and copy
        into ``buf`` under the GIL.
        """
        # Normalize to an unsigned-byte view. In MP mode the destination comes
        # straight from ``MemoryObj.byte_array`` (format "<B"); slice assignment
        # below requires a plain "B" view or it raises a structure mismatch.
        # The non-MP connector casts before calling; do it here so both callers
        # are safe.
        if buf.format != "B":
            buf = buf.cast("B")

        client = self._get_client()
        if self._has_buffer_get:
            size = buf.nbytes
            scratch = self._get_scratch(size)
            scratch_view = memoryview(scratch)[:size]
            result = client.get(key_str.encode(), buffer=scratch_view)
            if result is None:
                return False
            n = int(result)
            if n != size:
                # Fixed-size KV chunks must round-trip exactly. A short read
                # signals a size mismatch / corrupt or stale-format entry;
                # treat it as a miss rather than leaving stale tail bytes in
                # buf[n:] and reporting success.
                logger.warning(
                    "Valkey GET size mismatch for key %s: read %d bytes, "
                    "expected %d; treating as miss.",
                    key_str,
                    n,
                    size,
                )
                return False
            buf[:n] = scratch_view[:n]
            return True
        else:
            data = client.get(key_str.encode())
            if data is None:
                return False
            if len(data) != buf.nbytes:
                logger.warning(
                    "Valkey GET size mismatch for key %s: read %d bytes, "
                    "expected %d; treating as miss.",
                    key_str,
                    len(data),
                    buf.nbytes,
                )
                return False
            buf[: len(data)] = data
            return True

    def _do_exists(self, key_str: str) -> bool:
        """Check if a key exists (runs on a worker thread)."""
        return bool(self._get_client().exists([key_str.encode()]))

    def _do_delete(self, key_str: str) -> bool:
        """Delete a key (runs on a worker thread).

        Returns True if the key existed and was removed, False if it was
        absent. GLIDE ``delete`` returns the number of keys removed.
        """
        removed = self._get_client().delete([key_str.encode()])
        return bool(removed)

    def submit_set(self, key_str: str, data: bytes) -> Future:
        """Submit a SET operation."""
        return self._executor.submit(self._do_set, key_str, data)

    def submit_get_into(self, key_str: str, buf: memoryview) -> Future:
        """Submit a GET-into-buffer operation."""
        return self._executor.submit(self._do_get_into, key_str, buf)

    def submit_exists(self, key_str: str) -> Future:
        """Submit an EXISTS check."""
        return self._executor.submit(self._do_exists, key_str)

    def submit_delete(self, key_str: str) -> Future:
        """Submit a DELETE operation."""
        return self._executor.submit(self._do_delete, key_str)

    def _close_client(self) -> None:
        """Close the per-thread GLIDE client (runs on a worker thread)."""
        client = getattr(self._local, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                logger.debug("Error closing per-thread GLIDE client: %s", exc)
            self._local.client = None

    def close(self) -> None:
        """Shut down all per-thread GLIDE clients and the thread pool."""
        close_futs = [
            self._executor.submit(self._close_client) for _ in range(self.num_workers)
        ]
        for f in close_futs:
            try:
                f.result(timeout=self._request_timeout)
            except Exception as exc:
                logger.debug("Error during client close: %s", exc)
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Valkey thread pool closed")
