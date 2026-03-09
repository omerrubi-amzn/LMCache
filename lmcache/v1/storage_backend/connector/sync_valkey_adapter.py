# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class SyncValkeyConnectorAdapter(ConnectorAdapter):
    """Adapter for the SyncValkeyConnector (``valkey-sync://`` scheme).

    Uses the GLIDE sync client with a multiprocessing worker pool for
    high-throughput KV cache transfer. Requires ``valkey-glide`` with
    PRs #5492 and #5493.
    """

    def __init__(self) -> None:
        super().__init__("valkey-sync://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        """Create a SyncValkeyConnector from the given context.

        Args:
            context: Connector creation context containing URL, config,
                event loop, and local CPU backend.

        Returns:
            A configured SyncValkeyConnector instance.
        """
        # Local
        from .sync_valkey_connector import SyncValkeyConnector

        config = context.config
        extra_config = (
            config.extra_config
            if config is not None and config.extra_config is not None
            else {}
        )

        num_workers = int(extra_config.get("valkey_sync_num_workers", 8))
        username = str(extra_config.get("valkey_username", ""))
        password = str(extra_config.get("valkey_password", ""))
        raw_database_id = extra_config.get("valkey_database", None)
        database_id = int(raw_database_id) if raw_database_id is not None else None
        tls_enable = bool(extra_config.get("tls_enable", False))

        logger.info("Creating SyncValkey connector for URL: %s", context.url)
        parsed_url = parse_remote_url(context.url)
        return SyncValkeyConnector(
            host=parsed_url.host,
            port=parsed_url.port,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            num_workers=num_workers,
            username=username,
            password=password,
            database_id=database_id,
            tls_enable=tls_enable,
        )
