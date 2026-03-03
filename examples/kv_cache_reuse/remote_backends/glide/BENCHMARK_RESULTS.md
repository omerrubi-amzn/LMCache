# GLIDE Zero-Copy Optimization for LMCache ValkeyConnector

## Summary

We identified and eliminated unnecessary memory copies in the GLIDE Python sync client that were limiting LMCache's ValkeyConnector throughput to ~0.5–0.75 GB/s for SET/GET of large KV cache tensors. With two targeted changes — zero-copy SET via `memoryview` passthrough and zero-allocation GET via a new `get_into_buffer` FFI function — the GLIDE sync client now matches the C++ RESP client baseline (~1.1 GB/s), removing the need for a separate C++ dependency.

| Operation | Before | After | C++ RESP Baseline |
|-----------|--------|-------|-------------------|
| SET | 0.56 GB/s (async) / 0.97 GB/s (sync) | **1.03 GB/s** | 1.12 GB/s |
| GET | 0.75 GB/s | **1.15 GB/s** | 1.19 GB/s |

## Root Cause Analysis

### SET bottleneck: `bytes(memoryview)` copy

In `ValkeyConnector._put()`, the line:
```python
kv_bytes = bytes(memory_obj.byte_array)  # 32 MB malloc + memcpy per chunk
```
allocated a new `bytes` object from the tensor's `memoryview` before passing it to `set()`. This was the dominant cost — a 32 MB `malloc` + `memcpy` under the GIL for every chunk.

The GLIDE sync client's CFFI layer (`ffi.from_buffer()`) already supports any Python buffer-protocol object and extracts a raw pointer without copying. But the `isinstance` check in `_to_c_strings()` only accepted `bytes`, rejecting `memoryview`.

**Fix:** Change `isinstance(arg, bytes)` → `isinstance(arg, (bytes, bytearray, memoryview))` in two places in `glide_client.py`. This lets `memoryview` flow directly to `ffi.from_buffer()` — true zero-copy from tensor memory to the network.

### GET bottleneck: intermediate `bytes` allocation

The GET response path in the sync client:
```python
def _handle_string_response(self, msg):
    return self._ffi.buffer(msg.string_value, msg.string_value_len)[:]  # 32 MB copy
```
The `[:]` slice copies the C-owned response buffer into a new Python `bytes` object. LMCache then copies that `bytes` into the pre-allocated tensor buffer — two copies total.

**Fix:** New `command_get_into_buffer()` FFI function in Rust that receives a pointer to the caller's target buffer. After the Valkey response arrives as a `Value::BulkString(Vec<u8>)`, it copies directly from the Rust `Vec` into the Python `memoryview` — one copy instead of two, and no Python `bytes` allocation.

## Implementation Details

### 1. Zero-copy SET (isinstance patch)

**File:** `python/glide-sync/glide_sync/glide_client.py`

Two `isinstance` checks changed:
- `_to_c_strings()` (single command path, line 340)
- `_convert_commands_to_c_batch_info()` (batch command path, line 584)

```python
# Before:
elif isinstance(arg, bytes):
# After:
elif isinstance(arg, (bytes, bytearray, memoryview)):
```

This is Python-layer only. No changes to glide-core, no impact on other language clients.

### 2. Zero-alloc GET (`get_into_buffer`)

**Rust FFI** (`ffi/src/lib.rs`): New `command_get_into_buffer()` function that:
- Takes `target_buf: *mut u8` and `target_len: usize` from the caller
- Executes GET, receives `Value::BulkString(data)`
- Copies directly: `std::ptr::copy_nonoverlapping(data.as_ptr(), target_buf, copy_len)`
- Returns bytes written as `int_value` (or -1 for nil)

**Python CFFI** (`_glide_ffi.py`): Added FFI declaration for `command_get_into_buffer`.

**Python client** (`glide_client.py`): Added `get_into_buffer(key, buffer)` method on `BaseClient` that passes `ffi.from_buffer(buffer)` to the new FFI function.

### 3. LMCache ValkeyConnector change

**File:** `lmcache/v1/storage_backend/connector/valkey_connector.py`

Both `ValkeyConnector._put()` and `ValkeyClusterConnector._put()`:
```python
# Before:
kv_bytes = bytes(memory_obj.byte_array)
# After:
kv_bytes = memory_obj.byte_array
if not isinstance(kv_bytes, memoryview):
    kv_bytes = memoryview(kv_bytes)
elif kv_bytes.format != "B":
    kv_bytes = kv_bytes.cast("B")
```

## Why Sync, Not Async

The LMCache connector currently uses the async GLIDE client (`GlideClient`). We benchmarked both paths:

| Client | Architecture | SET (GB/s) | Bottleneck |
|--------|-------------|-----------|------------|
| Async GLIDE | PyO3 → protobuf → UDS → glide-core | 0.185 | Single UDS connection, protobuf serialization |
| Sync GLIDE | CFFI → direct FFI call → glide-core | **1.03** | Network bandwidth |

The async path is fundamentally limited by its single Unix Domain Socket connection and protobuf serialization overhead. The sync path uses direct CFFI calls with N independent client connections across threads, saturating the network.

To use the sync path in LMCache (which has an async interface), the connector would wrap sync calls in `asyncio.to_thread()` or a `ThreadPoolExecutor`.

## Benchmarking Setup

**Infrastructure:**
- ElastiCache: `cache.r7g.2xlarge` (non-TLS, single node)
- EC2: `c5.4xlarge` (16 vCPU, 32 GB RAM)
- Region: `us-west-2`
- Network: Same AZ, enhanced networking

**Parameters:**
- Chunk size: 32 MB (matches Llama-3.1-8B KV cache chunk with 256 tokens, 8 KV heads, 128 head dim, fp16)
- Number of keys: 32 (simulates 8K context)
- Workers/connections: 8
- Total data per run: ~1.1 GB

**Software:**
- Python 3.11
- valkey-glide (sync) with patched `.so`
- Rust 1.93.1, protoc 25.1
- `lmcache_redis` C++ RESP client for baseline

## Complete Benchmark Results

All modes measured in a single run for consistency:

```
==================================================================
LMCache Complete Benchmark: 32 MB x 32 keys = 1.1 GB
Workers: 8
==================================================================
Mode                                                 Throughput
------------------------------------------------------------------
SET: C++ RESP baseline                                  1.123 GB/s
GET: C++ RESP baseline                                  1.191 GB/s
SET: GLIDE sync bytes() copy                            0.971 GB/s
SET: GLIDE sync memoryview (zero-copy)                  1.031 GB/s
GET: GLIDE sync get() + copy                            0.753 GB/s
GET: GLIDE sync get_into_buffer (zero-alloc)            1.152 GB/s
==================================================================
```

### Async GLIDE Results (for reference)

```
Async GLIDE: 32 MB x 32 keys = 1.1 GB
Mode                                            SET (GB/s)   GET (GB/s)
-----------------------------------------------------------------------
A: batch + bytes(mv) [current connector]             0.148        0.316
B: individual set + bytes(mv)                        0.135        0.316
C: individual set_from_buffer (zero-copy)            0.185        0.317
```

## Action Items

### Completed

- [x] Root cause analysis of GLIDE vs C++ RESP throughput gap
- [x] Prototype and benchmark zero-copy SET (isinstance patch)
- [x] Prototype and benchmark zero-alloc GET (`get_into_buffer`)
- [x] PR submitted for isinstance patch: [omerrubi-amzn/valkey-glide#python/sync-accept-buffer-types](https://github.com/omerrubi-amzn/valkey-glide/tree/python/sync-accept-buffer-types)
- [x] LMCache `_put()` memoryview change implemented

### Next Steps

1. **Upstream PR #1 (isinstance patch)** — Submitted to `valkey-io/valkey-glide`. Small, safe, Python-only. Awaiting review.

2. **Upstream PR #2 (`get_into_buffer`)** — Commit on branch `python/sync-get-into-buffer`. Touches Rust FFI layer (`ffi/src/lib.rs`), Python CFFI declarations, and Python client. Needs tests before submitting.

3. **LMCache ValkeyConnector: switch to sync client** — The current connector uses async GLIDE. To realize the full throughput gains, it needs to switch to `glide_sync.GlideClient` with a thread pool. Options:
   - Add a `SyncValkeyConnector` variant behind the existing `RemoteConnector` interface
   - Or replace the async connector entirely

4. **LMCache ValkeyConnector: integrate `get_into_buffer`** — Replace the current `get()` + copy pattern with `get_into_buffer()` writing directly into the tensor's memory.

5. **Clean up AWS resources** — ElastiCache cluster, EC2 instance, security group, key pair, subnet group used for benchmarking.

## Files Modified

### valkey-glide

| File | Change |
|------|--------|
| `python/glide-sync/glide_sync/glide_client.py` | Accept `memoryview`/`bytearray` in `_to_c_strings()` and batch path; add `get_into_buffer()` |
| `python/glide-sync/glide_sync/_glide_ffi.py` | Add `command_get_into_buffer` FFI declaration |
| `ffi/src/lib.rs` | Add `command_get_into_buffer()` Rust FFI function |
| `python/tests/sync_tests/test_sync_client.py` | Add `test_sync_set_get_with_bytearray_and_memoryview` |
| `python/tests/sync_tests/test_sync_batch.py` | Add `test_sync_batch_set_with_bytearray_and_memoryview` |

### LMCache

| File | Change |
|------|--------|
| `lmcache/v1/storage_backend/connector/valkey_connector.py` | `_put()`: `bytes()` → `memoryview` in both connector classes |

### Benchmark Scripts

| File | Description |
|------|-------------|
| `benchmark_complete.py` | Combined GLIDE + C++ RESP benchmark (final) |
| `benchmark_zero_copy_sync.py` | Sync SET: bytes vs memoryview comparison |
| `benchmark_zero_copy_async.py` | Async SET: batch vs individual vs PyO3 zero-copy |
| `benchmark_get_into_buffer.py` | GET: regular vs get_into_buffer |
| `benchmark_glide_client.py` | Original 6-mode benchmark |
| `benchmark_resp_standalone.py` | C++ RESP baseline |
