Valkey (GLIDE)
==============

An L2 adapter backed by the **GLIDE** Valkey client (``valkey-glide``),
targeting **Valkey** or **Redis** servers in standalone or cluster mode. I/O is
dispatched through a pool of worker threads, each holding its own GLIDE sync
client, so round-trips overlap when GLIDE releases the GIL during its native
calls.

This is the MP-mode counterpart of the non-MP ``valkey://`` connector. It is
distinct from the :doc:`RESP <resp>` adapter: RESP uses a native C++ RESP
client, whereas this adapter uses the GLIDE client (zero-copy buffer GET when
available, cluster topology discovery, optional per-key TTL). Requires the
``valkey-glide`` package (>= 2.3); install with ``pip install valkey-glide``.

**Required fields:**

- ``host``: Valkey/Redis server hostname or IP.
- ``port``: Server port (positive integer).

**Optional fields:**

- ``num_workers`` (int, default ``8``): GLIDE I/O worker threads (> 0).
- ``username`` (string, default ``""``): Auth username.
- ``password`` (string, default ``""``): Auth password.
- ``tls_enable`` (bool, default ``false``): Use TLS.
- ``valkey_mode`` (string, default ``"standalone"``): ``"standalone"`` or
  ``"cluster"``.
- ``database_id`` (int, optional): Standalone DB id (ignored in cluster mode,
  which always uses DB 0).
- ``enable_ttl`` (bool, default ``false``): Write keys with a TTL so
  ``volatile-*`` eviction policies can reclaim them under memory pressure.
- ``ttl_sec`` (int, default ``86400``): TTL in seconds when ``enable_ttl`` is
  set (> 0).
- ``request_timeout`` (float, default ``5``): Per-request timeout (seconds).
- ``connection_timeout`` (float, default ``10``): Connect timeout (seconds).
- ``max_capacity_gb`` (float, default ``0``): Max L2 capacity in GB for usage
  tracking / aggregate eviction. ``0`` disables tracking.

When ``host``, ``port``, ``username``, or ``password`` are left empty, the
adapter falls back to the corresponding environment variables at creation
time: ``LMCACHE_VALKEY_HOST``, ``LMCACHE_VALKEY_PORT``,
``LMCACHE_VALKEY_USERNAME``, ``LMCACHE_VALKEY_PASSWORD``.

.. note::

   Like RESP, this adapter uses single-key, fixed-size storage with no
   per-chunk metadata, so partial/unfull chunks are not supported.

**Configuration examples:**

.. code-block:: bash

    # Basic standalone Valkey
    --l2-adapter '{"type": "valkey_glide", "host": "127.0.0.1", "port": 6379}'

    # Cluster mode with a per-key TTL and a capacity cap
    --l2-adapter '{"type": "valkey_glide", "host": "valkey.internal", "port": 6379, "valkey_mode": "cluster", "num_workers": 16, "enable_ttl": true, "ttl_sec": 86400, "max_capacity_gb": 50}'
