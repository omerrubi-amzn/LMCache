# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``ValkeyGlideL2AdapterConfig`` and the valkey_glide factory:
config ``from_dict`` validation, registry lookup/auto-discovery, and env-var
precedence in the factory. No ``glide_sync`` or Valkey server required — the
GLIDE pool and the native wrapper are monkeypatched.
"""

# Standard
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    get_l2_adapter_config_class,
    get_registered_l2_adapter_types,
)
from lmcache.v1.distributed.l2_adapters.valkey_glide_l2_adapter import (
    ValkeyGlideL2AdapterConfig,
    _create_valkey_glide_l2_adapter,
)

_MIN = {"type": "valkey_glide", "host": "h", "port": 6379}


def test_registered_and_discoverable():
    assert "valkey_glide" in get_registered_l2_adapter_types()
    assert get_l2_adapter_config_class("valkey_glide") is ValkeyGlideL2AdapterConfig


def test_from_dict_minimal_defaults():
    cfg = ValkeyGlideL2AdapterConfig.from_dict(_MIN)
    assert cfg.host == "h" and cfg.port == 6379
    assert cfg.num_workers == 8
    assert cfg.valkey_mode == "standalone"
    assert cfg.enable_ttl is False
    assert cfg.database_id is None
    assert cfg.max_capacity_gb == 0.0


def test_from_dict_full():
    cfg = ValkeyGlideL2AdapterConfig.from_dict(
        {
            **_MIN,
            "num_workers": 16,
            "username": "u",
            "password": "p",
            "tls_enable": True,
            "valkey_mode": "standalone",
            "database_id": 3,
            "enable_ttl": True,
            "ttl_sec": 120,
            "request_timeout": 2.5,
            "connection_timeout": 7,
            "max_capacity_gb": 4,
        }
    )
    assert cfg.num_workers == 16
    assert cfg.username == "u" and cfg.password == "p"
    assert cfg.tls_enable is True
    assert cfg.database_id == 3
    assert cfg.enable_ttl is True and cfg.ttl_sec == 120
    assert cfg.request_timeout == 2.5 and cfg.connection_timeout == 7.0
    assert cfg.max_capacity_gb == 4.0


def test_cluster_mode_ignores_database_id():
    cfg = ValkeyGlideL2AdapterConfig.from_dict(
        {**_MIN, "valkey_mode": "cluster", "database_id": 5}
    )
    assert cfg.valkey_mode == "cluster"
    assert cfg.database_id is None


@pytest.mark.parametrize(
    "patch, msg",
    [
        ({"host": ""}, "host"),
        ({"host": 123}, "host"),
        ({"port": 0}, "port"),
        ({"port": -1}, "port"),
        ({"port": True}, "port"),
        ({"port": "6379"}, "port"),
        ({"num_workers": 0}, "num_workers"),
        ({"num_workers": True}, "num_workers"),
        ({"valkey_mode": "sentinel"}, "valkey_mode"),
        ({"database_id": "x"}, "database_id"),
        ({"ttl_sec": 0}, "ttl_sec"),
        ({"ttl_sec": True}, "ttl_sec"),
        ({"request_timeout": 0}, "request_timeout"),
        ({"connection_timeout": -1}, "connection_timeout"),
        ({"max_capacity_gb": -1}, "max_capacity_gb"),
        ({"max_capacity_gb": True}, "max_capacity_gb"),
    ],
)
def test_from_dict_validation_errors(patch, msg):
    d = {**_MIN, **patch}
    with pytest.raises(ValueError, match=msg):
        ValkeyGlideL2AdapterConfig.from_dict(d)


def test_help_mentions_key_fields():
    h = ValkeyGlideL2AdapterConfig.help()
    for token in ("host", "port", "valkey_mode", "enable_ttl", "LMCACHE_VALKEY_HOST"):
        assert token in h


# ---------------------------------------------------------------------------
# Factory env-var precedence (monkeypatched pool + native wrapper)
# ---------------------------------------------------------------------------


class _RecordingPool:
    """Records the kwargs the factory passes; no real connection."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _RecordingPool.last_kwargs = kwargs

    def close(self):
        pass


class _StubAdapter:
    """Stub NativeConnectorL2Adapter capturing constructor args."""

    last = None

    def __init__(self, client, **kwargs):
        _StubAdapter.last = {"client": client, **kwargs}

    def close(self):
        pass


@pytest.fixture
def patched_factory(monkeypatch):
    # Ensure the lazy `import glide_sync` in the factory succeeds without the
    # real package being installed.
    monkeypatch.setitem(sys.modules, "glide_sync", type(sys)("glide_sync"))
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.connector.glide_pool._ThreadWorkerPool",
        _RecordingPool,
    )
    monkeypatch.setattr(
        "lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter."
        "NativeConnectorL2Adapter",
        _StubAdapter,
    )
    _RecordingPool.last_kwargs = {}
    _StubAdapter.last = None
    return monkeypatch


def test_config_values_take_precedence_over_env(patched_factory, monkeypatch):
    monkeypatch.setenv("LMCACHE_VALKEY_HOST", "envhost")
    monkeypatch.setenv("LMCACHE_VALKEY_PORT", "1111")
    monkeypatch.setenv("LMCACHE_VALKEY_USERNAME", "envuser")
    monkeypatch.setenv("LMCACHE_VALKEY_PASSWORD", "envpass")

    cfg = ValkeyGlideL2AdapterConfig.from_dict(
        {**_MIN, "host": "cfghost", "port": 2222, "username": "cfguser"}
    )
    _create_valkey_glide_l2_adapter(cfg)

    kw = _RecordingPool.last_kwargs
    assert kw["host"] == "cfghost"
    assert kw["port"] == 2222
    assert kw["username"] == "cfguser"
    # password empty in config -> env fallback
    assert kw["password"] == "envpass"


def test_env_used_when_config_empty(patched_factory, monkeypatch):
    monkeypatch.setenv("LMCACHE_VALKEY_USERNAME", "envuser")
    monkeypatch.setenv("LMCACHE_VALKEY_PASSWORD", "envpass")

    cfg = ValkeyGlideL2AdapterConfig.from_dict(_MIN)
    _create_valkey_glide_l2_adapter(cfg)

    kw = _RecordingPool.last_kwargs
    assert kw["username"] == "envuser"
    assert kw["password"] == "envpass"


def test_ttl_wired_only_when_enabled(patched_factory):
    cfg_off = ValkeyGlideL2AdapterConfig.from_dict(_MIN)
    _create_valkey_glide_l2_adapter(cfg_off)
    assert _RecordingPool.last_kwargs["ttl_seconds"] is None

    cfg_on = ValkeyGlideL2AdapterConfig.from_dict(
        {**_MIN, "enable_ttl": True, "ttl_sec": 99}
    )
    _create_valkey_glide_l2_adapter(cfg_on)
    assert _RecordingPool.last_kwargs["ttl_seconds"] == 99


def test_cluster_mode_flag_forwarded(patched_factory):
    cfg = ValkeyGlideL2AdapterConfig.from_dict({**_MIN, "valkey_mode": "cluster"})
    _create_valkey_glide_l2_adapter(cfg)
    assert _RecordingPool.last_kwargs["cluster_mode"] is True
    assert _StubAdapter.last["type_name"] == "valkey_glide"
