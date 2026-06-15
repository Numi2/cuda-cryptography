# cuda-cryptography

`cuda-cryptography` is a small CUDA/C++20 benchmark and correctness suite for
post-quantum and proof-system-adjacent cryptographic primitives.

This is research and benchmark code, not production cryptography. It has not
been audited, does not try to provide side-channel resistance, and should not be
used to protect real assets.

## Implemented Primitives

- finite-field arithmetic over the Goldilocks prime `p = 2^64 - 2^32 + 1`,
- field vector add and multiply,
- radix-2 NTT over the same field,
- Merkle tree root construction using SHA-256,
- ML-KEM/Kyber-style NTT polynomial multiplication over `q = 3329`.

Each primitive has a straightforward CPU reference implementation. CUDA builds
add GPU implementations and correctness tests that compare CPU output against
CUDA output.

## Why These Workloads

### Batched ML-KEM / ML-DSA

NVIDIA cuPQC directly targets ML-KEM and ML-DSA, including high-throughput
GPU public-key operations. Its public material reports H100 throughput for
ML-KEM-768 key generation, encapsulation, and decapsulation in millions of
operations per second, which makes batched PQC highly relevant to NVIDIA-facing
GPU cryptography work.

Source: [NVIDIA cuPQC](https://developer.nvidia.com/cupqc)

### Hashing + Merkle Forests

NVIDIA cuPQC highlights SHA-2, SHA-3, SHAKE, Poseidon2, and Merkle trees.
Its Merkle tree discussion emphasizes larger tree sizes, which matches the
CUDA story: more leaves expose more parallelism and make GPU residency matter.

Source: [NVIDIA cuPQC Hash/Merkle](https://developer.nvidia.com/cupqc)

### ZK Proof Kernels: MSM + NTT

ZK GPU performance research identifies MSM and NTT as the dominant prover
kernels; the cited paper notes that other kernels are under 5% of total time in
its breakdown. It also calls out NTT bottlenecks at scale and overheads from
CPU-GPU transfers and repeated kernel launches. This project therefore focuses
on batching, device residency, and NTT/Merkle kernels rather than isolated tiny
operations.

Source: [ZKProphet GPU ZKP performance](https://arxiv.org/html/2509.22684v1)

### Large Batched NTT for Lattice/FHE/PQC

NTT is core to Kyber/ML-KEM, Dilithium/ML-DSA, and many FHE/proof-system
workloads. A single Kyber-sized `n = 256` polynomial is too small to make CUDA
look good by itself, so the benchmark headline is batched ML-KEM-style NTT and
polynomial multiplication at `1`, `1K`, `10K`, and `100K` polynomial batches.

Source: [Cambridge/OpenTitan NTT acceleration paper](https://www.cl.cam.ac.uk/~fms27/papers/2024-UrquhartStajano-acceleration.pdf)

## ML-KEM/Kyber Primitive Path

This module is an ML-KEM/Kyber-style primitive benchmark around hot kernels,
not a complete ML-KEM implementation. It uses:

- modulus `q = 3329`,
- degree `n = 256`,
- primitive 256th root `17`,
- CPU schoolbook polynomial multiplication as the obvious reference,
- CPU NTT polynomial multiplication,
- CUDA batched NTT and batched NTT-based polynomial multiplication.

The benchmark includes:

- CPU ML-KEM NTT `n=256`,
- CUDA ML-KEM NTT `batch=1`,
- CUDA ML-KEM NTT `batch=1024`, `batch=10000`, and `batch=100000`,
- CUDA ML-KEM poly mul `batch=1`,
- CUDA ML-KEM poly mul `batch=1024`, `batch=10000`, and `batch=100000`.

The single-polynomial CUDA row is intentionally included to show launch and
transfer overhead. The meaningful GPU story is the batched path.

This is not production cryptography. It is not a constant-time, audited, or
standards-complete ML-KEM implementation.

## Recommended Next Addition: Poseidon2 Merkle Forests

The strongest next module for `cuda-cryptography` is `poseidon2_merkle_forest`.
It aligns with NVIDIA cuPQC's public direction around GPU hash, Merkle, and
Poseidon2 workloads, fits ZK/proof-system roles better than plain SHA-256, and
extends the existing field and Merkle code instead of starting from scratch.

Target design:

- Poseidon2 hash over BabyBear or Goldilocks field,
- CPU reference implementation,
- CUDA implementation,
- batched Merkle tree builder for one tree with `2^20` leaves,
- batched Merkle forest builder for `1024` trees with `2^10` leaves,
- `65536` small authentication trees,
- leaves and intermediate levels kept GPU-resident,
- CPU vs CUDA benchmark with host-transfer-included timing,
- device-only timing,
- throughput in hashes/sec and GB/s,
- proof path generation for many leaves,
- CPU verification of generated authentication paths.

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

## Validated CUDA Results: Goldilocks/SHA-256 Path

The original Goldilocks NTT, field arithmetic, and SHA-256 Merkle path was
validated on an NVIDIA RTX A6000 using CUDA 12.4:

- CUDA build passed,
- correctness tests passed,
- commit tested: `823b389`.

Optimization results from the portable CUDA baseline to the current
Goldilocks-folding and NTT-twiddle implementation:

| Primitive | Size | Baseline | Current | Change |
|-----------|-----:|---------:|--------:|-------:|
| NTT | 4096 | 4.32 Mops/s | 10.90 Mops/s | 2.52x |
| NTT | 16384 | 12.57 Mops/s | 31.25 Mops/s | 2.49x |
| Field mul | 65536 | 84.91 Mops/s | 94.82 Mops/s | 1.12x |
| Merkle SHA-256 | 16384 | 39.97 Mops/s | 41.70 Mops/s | 1.04x |

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
- CUDA field multiplication uses `__umul64hi` plus Goldilocks modular folding,
- NTT launches one kernel per transform stage,
- Merkle tree hashing launches one kernel per tree level,
- CUDA SHA-256 is implemented locally for fixed-size leaf and parent inputs.

This keeps the code readable for review while still showing the core GPU
engineering pieces: kernels, memory transfers, deterministic tests, and
benchmark reporting.

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

- fused NTT stages and shared-memory tiling,
- batched Merkle trees and persistent device buffers,
- CUDA events for device-only timing separate from host transfer timing,
- larger benchmark matrices by GPU architecture,
- comparison against established cryptography/proof-system libraries.
