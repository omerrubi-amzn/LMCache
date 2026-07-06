# SPDX-License-Identifier: Apache-2.0
"""
Valkey GLIDE L2 adapter config and factory (MP mode).

Backed by the pure-Python GLIDE ``_ThreadWorkerPool`` wrapped by
``GlideBatchClient`` (batch/eventfd shim) and ``NativeConnectorL2Adapter``.

This is the MP-mode counterpart of the non-MP ``ValkeyConnector``
(``valkey://`` scheme). It mirrors ``resp_l2_adapter.py``; the only difference
is that the backing client is Python (GLIDE) rather than the native C++
``LMCacheRedisClient``. See ``glide_batch_client.py`` for the adapter shim.

Registered under the type name ``"valkey_glide"`` and selected via::

    --l2-adapter '{"type":"valkey_glide","host":"127.0.0.1","port":6379}'
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Optional
import os

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory

logger = init_logger(__name__)

#: Default per-key TTL (seconds) when ``enable_ttl`` is set without ``ttl_sec``.
_DEFAULT_TTL_SECS = 86400


class ValkeyGlideL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the GLIDE Valkey client.

    Fields:
    - host: server hostname or IP (required).
    - port: server port (required, >0).
    - num_workers: GLIDE worker threads for I/O (default 8, >0).
    - username / password: optional auth.
    - tls_enable: use TLS (default False).
    - valkey_mode: "standalone" (default) or "cluster".
    - database_id: standalone DB id (ignored in cluster mode).
    - enable_ttl: write keys with a TTL so volatile-* eviction can reclaim
      them (default False).
    - ttl_sec: TTL in seconds when enable_ttl is set (default 86400, >0).
    - request_timeout / connection_timeout: GLIDE timeouts in seconds.
    - max_capacity_gb: L2 capacity for usage tracking / eviction
      (default 0 = disabled).
    """

    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int = 8,
        username: str = "",
        password: str = "",
        tls_enable: bool = False,
        valkey_mode: str = "standalone",
        database_id: Optional[int] = None,
        enable_ttl: bool = False,
        ttl_sec: int = _DEFAULT_TTL_SECS,
        request_timeout: float = 5.0,
        connection_timeout: float = 10.0,
        max_capacity_gb: float = 0,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.username = username
        self.password = password
        self.tls_enable = tls_enable
        self.valkey_mode = valkey_mode
        self.database_id = database_id
        self.enable_ttl = enable_ttl
        self.ttl_sec = ttl_sec
        self.request_timeout = request_timeout
        self.connection_timeout = connection_timeout
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "ValkeyGlideL2AdapterConfig":
        """Construct and validate a config from a raw dict (JSON/CLI)."""
        host = d.get("host")
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")

        port = d.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or port <= 0:
            raise ValueError("port must be a positive integer")

        num_workers = d.get("num_workers", 8)
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers <= 0
        ):
            raise ValueError("num_workers must be a positive integer")

        username = str(d.get("username", ""))
        password = str(d.get("password", ""))
        tls_enable = bool(d.get("tls_enable", False))

        valkey_mode = d.get("valkey_mode", "standalone")
        if valkey_mode not in ("standalone", "cluster"):
            raise ValueError("valkey_mode must be 'standalone' or 'cluster'")

        database_id = d.get("database_id", None)
        if database_id is not None:
            if isinstance(database_id, bool) or not isinstance(database_id, int):
                raise ValueError("database_id must be an integer or omitted")
            if valkey_mode == "cluster":
                logger.warning(
                    "database_id=%s is ignored in cluster mode "
                    "(Valkey cluster always uses DB 0).",
                    database_id,
                )
                database_id = None

        enable_ttl = bool(d.get("enable_ttl", False))
        ttl_sec = d.get("ttl_sec", _DEFAULT_TTL_SECS)
        if isinstance(ttl_sec, bool) or not isinstance(ttl_sec, int) or ttl_sec <= 0:
            raise ValueError("ttl_sec must be a positive integer number of seconds")

        request_timeout = d.get("request_timeout", 5.0)
        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, (int, float))
            or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be a positive number")

        connection_timeout = d.get("connection_timeout", 10.0)
        if (
            isinstance(connection_timeout, bool)
            or not isinstance(connection_timeout, (int, float))
            or connection_timeout <= 0
        ):
            raise ValueError("connection_timeout must be a positive number")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if isinstance(max_capacity_gb, bool) or not isinstance(
            max_capacity_gb, (int, float)
        ):
            raise ValueError("max_capacity_gb must be a non-negative number")
        if max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        return cls(
            host=host,
            port=port,
            num_workers=num_workers,
            username=username,
            password=password,
            tls_enable=tls_enable,
            valkey_mode=str(valkey_mode),
            database_id=database_id,
            enable_ttl=enable_ttl,
            ttl_sec=ttl_sec,
            request_timeout=float(request_timeout),
            connection_timeout=float(connection_timeout),
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Valkey GLIDE L2 adapter config fields:\n"
            "- host (str): Valkey server hostname or IP (required)\n"
            "- port (int): server port (required, >0)\n"
            "- num_workers (int): GLIDE I/O worker threads (default 8, >0)\n"
            "- username / password (str): optional auth (default empty)\n"
            "- tls_enable (bool): use TLS (default false)\n"
            "- valkey_mode (str): 'standalone' (default) or 'cluster'\n"
            "- database_id (int): standalone DB id (ignored in cluster)\n"
            "- enable_ttl (bool): write keys with a TTL for volatile-* "
            "eviction (default false)\n"
            "- ttl_sec (int): TTL in seconds when enable_ttl is set "
            "(default 86400, >0)\n"
            "- request_timeout (float): request timeout seconds (default 5)\n"
            "- connection_timeout (float): connect timeout seconds (default 10)\n"
            "- max_capacity_gb (float): L2 capacity in GB for usage tracking / "
            "eviction (default 0 = disabled)\n\n"
            "Environment variable defaults (used when the config value is empty, "
            "read at adapter creation, not stored in config):\n"
            "- LMCACHE_VALKEY_HOST: default host\n"
            "- LMCACHE_VALKEY_PORT: default port\n"
            "- LMCACHE_VALKEY_USERNAME: default username\n"
            "- LMCACHE_VALKEY_PASSWORD: default password"
        )


def _create_valkey_glide_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the GLIDE batch client."""
    del l1_memory_desc

    try:
        # Third Party
        import glide_sync  # type: ignore[import-untyped] # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Valkey GLIDE L2 adapter requires the 'valkey-glide' package "
            "(>=2.3). Install with: pip install valkey-glide"
        ) from e

    # Lazy imports to avoid import-time cost / circular deps.
    # First Party
    from lmcache.v1.distributed.l2_adapters.glide_batch_client import GlideBatchClient
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
        NativeConnectorL2Adapter,
    )
    from lmcache.v1.storage_backend.connector.glide_pool import _ThreadWorkerPool

    assert isinstance(config, ValkeyGlideL2AdapterConfig)

    # Config/CLI args take precedence over environment variables, which serve
    # as defaults. This keeps secrets out of logged config while allowing
    # explicit CLI overrides (matches the RESP adapter).
    host = config.host or os.environ.get("LMCACHE_VALKEY_HOST", "")
    port = (
        config.port if config.port else int(os.environ.get("LMCACHE_VALKEY_PORT", "0"))
    )
    username = config.username or os.environ.get("LMCACHE_VALKEY_USERNAME", "")
    password = config.password or os.environ.get("LMCACHE_VALKEY_PASSWORD", "")

    if not host:
        raise ValueError("host must be a non-empty string")
    if not port:
        raise ValueError("port must be a positive integer")

    cluster_mode = config.valkey_mode == "cluster"
    ttl_seconds = config.ttl_sec if config.enable_ttl else None

    pool = _ThreadWorkerPool(
        host=host,
        port=port,
        num_workers=config.num_workers,
        username=username,
        password=password,
        request_timeout=config.request_timeout,
        connection_timeout=config.connection_timeout,
        tls_enable=config.tls_enable,
        cluster_mode=cluster_mode,
        database_id=config.database_id,
        ttl_seconds=ttl_seconds,
    )
    client = GlideBatchClient(pool)

    logger.info(
        "Created Valkey GLIDE L2 adapter: %s:%d (mode=%s, workers=%d, ttl=%s)",
        host,
        port,
        config.valkey_mode,
        config.num_workers,
        ttl_seconds,
    )
    return NativeConnectorL2Adapter(
        client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="valkey_glide",
        extra_status={
            "host": host,
            "port": port,
            "valkey_mode": config.valkey_mode,
            "num_workers": config.num_workers,
            "ttl_seconds": ttl_seconds,
        },
    )


# Self-register config type and adapter factory (auto-discovered by
# l2_adapters/factory.py via pkgutil on first use).
register_l2_adapter_type("valkey_glide", ValkeyGlideL2AdapterConfig)
register_l2_adapter_factory("valkey_glide", _create_valkey_glide_l2_adapter)
