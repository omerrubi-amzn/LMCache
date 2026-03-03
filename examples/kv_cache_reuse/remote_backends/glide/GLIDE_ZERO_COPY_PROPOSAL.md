# Proposal: Zero-Copy Buffer API for Valkey GLIDE Python Client

## Summary

We propose adding a zero-copy buffer API to valkey-glide's Python client that
allows callers to SET/GET large binary values directly from/into pre-allocated
`memoryview` buffers without intermediate Python `bytes` object construction.
This eliminates the primary throughput bottleneck for large-value workloads
such as LLM KV cache storage, where individual values are 32–80 MB.

## Motivation

### The use case

[LMCache](https://github.com/LMCache/LMCache) is an open-source KV cache
management system for LLM inference. It stores attention state (key-value
vectors) in a remote Valkey/Redis backend for cross-instance sharing. Each
cache chunk is 32 MB (Llama-3.1-8B) to 80 MB (Llama-3.1-70B), and a single
request may store/retrieve 32–256 chunks.

### The bottleneck

We benchmarked GLIDE against LMCache's C++ RESP client on an ElastiCache
`cache.r7g.2xlarge` instance with the `llama8b-8k` workload (32 MB × 32 keys
= 1 GB total):

| Client                          | SET (GB/s) | GET (GB/s) |
|---------------------------------|-----------|-----------|
| C++ RESP (8 threads, 8 sockets) | 1.05      | 1.07      |
| GLIDE sync, N clients (test 6)  | 0.56      | 0.50      |
| GLIDE async pipeline (test 4)   | 0.24      | 0.24      |
| GLIDE async individual (test 1) | 0.20      | 0.23      |

The best GLIDE configuration reaches ~50% of the C++ baseline. Profiling
shows the gap is dominated by Python-side data marshalling:

1. **SET path**: The caller must construct a `bytes` object (`bytes(memoryview)`)
   before passing it to `client.set()`. For a 32 MB value, this is a 32 MB
   `malloc` + `memcpy` under the GIL, taking ~10 ms per chunk.

2. **GET path**: GLIDE returns a `bytes` object allocated in Rust and copied
   into Python memory. The caller then copies it into the target buffer
   (`target_view[:] = result`). This is another 32 MB copy under the GIL.

3. **Protobuf serialization**: The value bytes are serialized into a protobuf
   message for the UDS transport between Python and the Rust core, adding
   another copy.

The C++ RESP client avoids all of this: it extracts a raw pointer from the
Python `memoryview` once (under the GIL), then performs all socket I/O in C++
threads with zero additional copies.

### Why this matters beyond LMCache

Any application storing large binary blobs in Valkey (ML model weights,
embeddings, image/video data, serialized tensors) hits the same bottleneck.
The current API forces at least two full copies of the value through Python
heap memory, which is both slow and memory-wasteful for multi-MB values.

## Proposed API

### Python surface

```python
# ── SET from buffer (zero-copy) ─────────────────────────────

# New method: writes directly from a buffer to the socket
await client.set_from_buffer(
    key: TEncodable,
    buffer: memoryview,              # source data, read-only access
    header: Optional[bytes] = None,  # optional prefix prepended to value
)

# Batch variant
batch.set_from_buffer(key, buffer, header=None)
await client.exec(batch)

# ── GET into buffer (zero-copy) ─────────────────────────────

# New method: reads directly from socket into a pre-allocated buffer
bytes_read = await client.get_into_buffer(
    key: TEncodable,
    buffer: memoryview,  # target buffer, must be writable and large enough
) -> Optional[int]       # returns bytes written, or None if key missing
```

### Semantics

- `set_from_buffer(key, buf, header=b"HDR")` is equivalent to
  `set(key, b"HDR" + bytes(buf))` but without constructing the concatenated
  `bytes` object in Python.

- `get_into_buffer(key, buf)` is equivalent to
  `result = get(key); buf[:len(result)] = result` but without allocating the
  intermediate `bytes` object.

- The `header` parameter supports the common pattern of prepending a small
  metadata struct (e.g., tensor shapes/dtypes) to a large binary payload.
  This is how LMCache's ValkeyConnector packs metadata + KV bytes into a
  single key.

- The caller guarantees the buffer remains valid for the duration of the
  async operation (same lifetime contract as the existing `bytes` API).

### Sync client

The same API applies to `glide_sync`:

```python
sync_client.set_from_buffer(key, buffer, header=None)
bytes_read = sync_client.get_into_buffer(key, buffer)
```

## Implementation Design

### Current data path (the problem)

```
Python                              Rust (PyO3)                    Rust Core
──────                              ───────────                    ─────────
set(key, bytes(memoryview))         create_leaked_bytes_vec()      deserialize protobuf
  │                                   │                              │
  ├─ COPY 1: bytes(mv)               ├─ COPY 2: bytes.to_vec()      ├─ COPY 3: protobuf
  │  malloc+memcpy 32MB              │  into Rust Vec<u8>            │  decode into Bytes
  │  under GIL                       │  (lib.rs:522)                 │
  │                                   │                              ├─ Build RESP frame
  ├─ Encode into protobuf            ├─ Leak as Box<Vec<Bytes>>      ├─ Write to socket
  │  (value embedded in msg)          │  pass pointer via protobuf   │
  └─ Send over UDS                   └─                             └─
```

Three copies of a 32 MB value before it reaches the socket. The `bytes(memoryview)`
in Python and `bytes.to_vec()` in Rust are the dominant costs.

### Proposed data path (zero-copy)

```
Python                              Rust (PyO3)                    Rust Core
──────                              ───────────                    ─────────
set_from_buffer(key, memoryview)    PyBuffer<u8> extraction        Receive buffer descriptor
  │                                   │                              │
  ├─ Extract ptr+len from mv         ├─ NO COPY: get raw ptr+len    ├─ Build RESP frame header
  │  (under GIL, ~1μs)               │  from PyBuffer                │
  │                                   │                              ├─ writev() to socket:
  ├─ Send buffer descriptor           ├─ Pass (ptr, len) as u64s     │  [RESP header]
  │  via protobuf (just 16 bytes)     │  in protobuf message         │  [header bytes from ptr]
  │                                   │                              │  [buffer bytes from ptr]
  └─ Await completion                └─ Release GIL                 └─ Signal done
```

Zero copies between Python and the socket. The Rust core reads directly from
the Python buffer's memory address.

### Changes by layer

#### 1. Protobuf (`glide-core/src/protobuf/command_request.proto`)

Add a buffer descriptor message and extend `Command.args`:

```protobuf
message BufferArg {
    uint64 pointer = 1;    // raw pointer to buffer data
    uint64 length = 2;     // buffer length in bytes
    bytes header = 3;      // optional small header to prepend
}

message Command {
    message ArgsArray {
        repeated bytes args = 1;
    }

    RequestType request_type = 1;
    oneof args {
        ArgsArray args_array = 2;
        uint64 args_vec_pointer = 3;
        BufferArg buffer_arg = 4;       // NEW: zero-copy buffer path
    }
}
```

#### 2. PyO3 bindings (`python/glide-async/src/lib.rs`)

Add a function that extracts a `PyBuffer` pointer without copying:

```rust
#[pyfunction]
pub fn create_leaked_buffer_arg(
    py: Python,
    key: Bound<PyBytes>,
    buffer: PyBuffer<u8>,
    header: Option<Bound<PyBytes>>,
) -> (usize, usize, usize, Option<Vec<u8>>) {
    // Extract raw pointer and length — NO COPY
    let ptr = buffer.buf_ptr() as usize;
    let len = buffer.len_bytes();

    // Key still needs to be copied (small, ~100 bytes)
    let key_bytes = Bytes::from(key.as_bytes().to_vec());
    let key_ptr = from_mut(Box::leak(Box::new(key_bytes))) as usize;

    let hdr = header.map(|h| h.as_bytes().to_vec());
    (key_ptr, ptr, len, hdr)
}
```

#### 3. Python client (`python/glide-async/python/glide/glide_client.py`)

Add `set_from_buffer` that builds a `CommandRequest` with `buffer_arg`:

```python
async def set_from_buffer(
    self,
    key: TEncodable,
    buffer: memoryview,
    header: Optional[bytes] = None,
) -> Optional[bytes]:
    request = CommandRequest()
    request.callback_idx = self._get_callback_index()
    request.single_command.request_type = RequestType.Set
    # Pass buffer descriptor instead of copying value
    buf_info = buffer.cast('B')  # ensure byte-addressable
    request.single_command.buffer_arg.pointer = buf_info.obj.__array_interface__['data'][0]  # ctypes approach
    request.single_command.buffer_arg.length = len(buf_info)
    if header:
        request.single_command.buffer_arg.header = header
    return await self._write_request_await_response(request)
```

#### 4. Rust core (`glide-core/src/client/`)

Handle `buffer_arg` in the command dispatcher:

```rust
// When processing a SET with buffer_arg:
fn build_set_from_buffer(key: &[u8], buf_ptr: u64, buf_len: u64, header: &[u8]) -> Vec<IoSlice> {
    let value_len = header.len() + buf_len as usize;
    let resp_header = format!("*3\r\n$3\r\nSET\r\n${}\r\n{}\r\n${}\r\n",
        key.len(), std::str::from_utf8(key).unwrap(), value_len);

    // Use writev / scatter-gather to write without concatenating:
    let buf_slice = unsafe { std::slice::from_raw_parts(buf_ptr as *const u8, buf_len as usize) };
    vec![
        IoSlice::new(resp_header.as_bytes()),
        IoSlice::new(header),
        IoSlice::new(buf_slice),
        IoSlice::new(b"\r\n"),
    ]
}
```

For GET, the inverse — read the RESP bulk string response directly into
the target buffer instead of allocating a `Vec<u8>`.

#### 5. CFFI layer for glide-sync (`ffi/src/lib.rs`)

Expose equivalent C functions that accept buffer pointers:

```rust
#[no_mangle]
pub extern "C" fn glide_set_from_buffer(
    client: *mut GlideClient,
    key_ptr: *const u8, key_len: usize,
    buf_ptr: *const u8, buf_len: usize,
    hdr_ptr: *const u8, hdr_len: usize,
    callback: extern "C" fn(usize, *const u8, usize),
    callback_id: usize,
);
```

The Python `glide_sync` wrapper passes `ffi.from_buffer(memoryview)`.

### Safety considerations

- **Buffer lifetime**: The Python caller must keep the buffer alive until
  the operation completes. This is the same contract as `asyncio` buffer
  protocols and the existing `bytes` API (Python GC won't collect a
  referenced `memoryview`).
- **Pointer validity**: PyO3's `PyBuffer` validates the buffer is
  contiguous and byte-addressable before extracting the pointer.
- **Thread safety**: The buffer is accessed read-only for SET and
  exclusively for GET. The caller must not mutate the buffer during the
  operation (standard `memoryview` contract).

## Expected Impact

Based on our benchmarks, eliminating the two copies (Python→Rust for SET,
Rust→Python for GET) should:

- **SET**: Remove ~10 ms/chunk overhead (32 MB copy), bringing GLIDE
  throughput from 0.56 GB/s to ~0.9–1.0 GB/s (approaching C++ RESP).
- **GET**: Remove the intermediate `bytes` allocation and copy, bringing
  throughput from 0.50 GB/s to ~0.8–1.0 GB/s.
- **Memory**: Eliminate transient 32–80 MB Python `bytes` objects per
  operation, reducing peak memory by ~2× for large-value workloads.

## Action Items

### Phase 1: Prototype and validate (LMCache team)

- [ ] Fork `valkey-glide` and implement `set_from_buffer` in the PyO3 layer
      using `PyBuffer<u8>` to validate the zero-copy path works end-to-end.
- [ ] Benchmark the prototype against the C++ RESP baseline to confirm the
      expected throughput improvement.
- [ ] Document the buffer lifetime contract and safety invariants.

### Phase 2: Upstream proposal (LMCache → GLIDE team)

- [ ] Open a GitHub issue on `valkey-io/valkey-glide` with this proposal,
      benchmark data, and the prototype branch.
- [ ] Engage with GLIDE maintainers on API design — specifically whether
      this should be new methods (`set_from_buffer`) or an overload of
      existing `set`/`get` that detects `memoryview` input.
- [ ] Discuss whether the protobuf transport should carry buffer descriptors
      or if the UDS protocol needs a separate "buffer passthrough" mode.

### Phase 3: Implementation (joint with GLIDE team)

- [ ] Implement `set_from_buffer` / `get_into_buffer` in Rust core with
      scatter-gather socket writes.
- [ ] Add `Batch` support for buffer operations.
- [ ] Implement the `glide_sync` (CFFI) equivalent.
- [ ] Add integration tests with large values (>1 MB).
- [ ] Update documentation and migration guides.

### Phase 4: LMCache integration

- [ ] Update `ValkeyConnector` to use `set_from_buffer` / `get_into_buffer`
      when available (with fallback to current `bytes` path for older GLIDE).
- [ ] Re-run end-to-end benchmarks with the improved connector.
- [ ] Publish updated benchmark results.

## References

- [valkey-glide Python developer docs](https://glide.valkey.io/languages/python/developer/)
- [PyO3 buffer protocol](https://pyo3.rs/v0.22.0/types#buffer-protocol)
- [GLIDE architecture: Rust core](https://glide.valkey.io/concepts/architecture/rust-core-design/)
- [LMCache ValkeyConnector PR #1743](https://github.com/LMCache/LMCache/pull/1743)
- Benchmark data: `examples/kv_cache_reuse/remote_backends/glide/`
