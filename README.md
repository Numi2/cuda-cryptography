# cuda-pqc-bench

`cuda-pqc-bench` is a small CUDA/C++20 benchmark and correctness suite for
post-quantum and proof-system-adjacent cryptographic primitives.

This is research and benchmark code, not production cryptography. It has not
been audited, does not try to provide side-channel resistance, and should not be
used to protect real assets.

## Implemented Primitives

- finite-field arithmetic over the Goldilocks prime `p = 2^64 - 2^32 + 1`,
- field vector add and multiply,
- radix-2 NTT over the same field,
- Merkle tree root construction using SHA-256.

Each primitive has a straightforward CPU reference implementation. CUDA builds
add GPU implementations and correctness tests that compare CPU output against
CUDA output.

## Build

Requirements:

- CMake 3.24 or newer,
- a C++20 compiler,
- NVIDIA CUDA Toolkit for GPU targets.

Build with CUDA when `nvcc` is available:

```bash
./scripts/build.sh
```

Build CPU-only, useful on development machines or CI runners without CUDA:

```bash
./scripts/build.sh -DCPB_ENABLE_CUDA=OFF
```

## Test

```bash
./scripts/test.sh
```

The test executable checks deterministic field, SHA-256, NTT, and Merkle test
vectors. CUDA builds additionally compare CPU and CUDA outputs and print the
first mismatch index or byte on failure.

## Benchmark

```bash
./scripts/bench.sh
```

The benchmark prints elapsed time and throughput for several input sizes. GPU
rows are shown only when the project is built with CUDA and a CUDA device is
visible.

Sample output shape:

| Backend | Primitive     |        n |  Time (ms) | Throughput Mops/s |
|--------:|---------------|---------:|-----------:|-----------------:|
| CPU     | field add     |     1024 |      0.004 |           256.00 |
| CPU     | field mul     |     1024 |      0.018 |            56.89 |
| CUDA    | field add     |     1024 |      0.090 |            11.38 |
| CUDA    | radix-2 ntt   |     4096 |      1.700 |             2.41 |
| CUDA    | merkle sha256 |     4096 |      0.950 |             4.31 |

Actual numbers depend heavily on GPU model, CUDA version, clock state, and PCIe
transfer overhead.

## Architecture Notes

The field modulus is the Goldilocks prime:

```text
p = 18446744069414584321 = 2^64 - 2^32 + 1
```

It is convenient for proof-system benchmarks because field elements fit in
64-bit storage and `p - 1` is divisible by `2^32`, enabling large radix-2 NTTs.
The implementation uses primitive root `7`.

The CUDA kernels are intentionally simple:

- vector kernels map one field element to one CUDA thread,
- CUDA field multiplication uses a simple portable double-and-add reduction,
- NTT launches one kernel per transform stage,
- Merkle tree hashing launches one kernel per tree level,
- CUDA SHA-256 is implemented locally for fixed-size leaf and parent inputs.

This keeps the code readable for review and recruitment while still showing the
core GPU engineering pieces: kernels, memory transfers, deterministic tests,
and benchmark reporting.

## Repository Layout

```text
include/   public headers
src/       CPU and CUDA implementations
tests/     correctness test executable
bench/     benchmark executable
scripts/   build, test, and benchmark helpers
docs/      design notes
```

## Future Work

- optimized Goldilocks reduction without relying on 128-bit device arithmetic,
- fused NTT stages and shared-memory tiling,
- batched Merkle trees and persistent device buffers,
- CUDA events for device-only timing separate from host transfer timing,
- larger benchmark matrices by GPU architecture,
- comparison against established cryptography/proof-system libraries.
