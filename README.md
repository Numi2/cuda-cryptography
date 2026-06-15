# cuda-cryptography

CUDA/C++20 benchmark and correctness suite for post-quantum and
proof-system-adjacent cryptographic kernels.

This repository is intentionally focused: CPU references, CUDA kernels,
deterministic correctness tests, CMake, CI, and benchmark output for a small set
of workloads that matter for PQC and proof systems.

## Security Notice

This is research and benchmark code, not production cryptography.

The implementations are not audited, not constant-time, and not suitable for
protecting real assets. The Poseidon2 path is a Poseidon2-style Goldilocks
permutation with deterministic benchmark constants; it is not a standardized
Poseidon2 parameter set.

## What This Demonstrates

- CUDA kernels for field arithmetic, NTTs, SHA-256 Merkle trees, ML-KEM-style
  polynomial operations, and Poseidon2-style Merkle forests.
- Straightforward CPU reference implementations used for correctness checks.
- Deterministic vectors for field math, SHA-256, ML-KEM-style polynomials, and
  Poseidon2-style leaf/pair/root hashes.
- Batching for small PQC polynomials, because one Kyber-sized polynomial is too
  small to be a useful GPU workload.
- Device-resident Poseidon2-style Merkle forest levels, with roots copied back
  at the end.
- Benchmark reporting that separates host-transfer-included timing from
  Poseidon2 device-only CUDA event timing.

## Status

| Module | Status | Current Limitations |
|--------|--------|---------------------|
| Goldilocks field + radix-2 NTT | Implemented | Readable stage kernels, not a fully fused production NTT |
| SHA-256 Merkle tree | Implemented | Root builder only; no GPU auth-path extraction |
| Batched ML-KEM/Kyber primitive path | Implemented benchmark primitive | Not full ML-KEM keygen/encap/decap |
| Poseidon2-style Merkle forest | Experimental benchmark path | Benchmark constants, not standardized Poseidon2 parameters |

## Implemented Workloads

### Goldilocks Field + NTT

- field: `p = 2^64 - 2^32 + 1`
- CPU field add/mul reference
- CUDA field add/mul kernels
- radix-2 NTT and inverse NTT
- CPU-vs-CUDA correctness tests

### SHA-256 Merkle Tree

- local SHA-256 implementation
- CPU Merkle root builder
- CUDA leaf and parent hashing kernels
- deterministic CPU-vs-CUDA root checks

### Batched ML-KEM/Kyber Primitive Path

- modulus: `q = 3329`
- degree: `n = 256`
- primitive 256th root: `17`
- CPU schoolbook polynomial multiplication
- CPU NTT-based polynomial multiplication
- CUDA batched NTT
- CUDA batched NTT-based polynomial multiplication
- benchmark batches in full mode: `1`, `1024`, `10000`, `100000`

The point is batching. A single Kyber-sized polynomial is too small to make CUDA
look good; batched polynomial operations are the relevant GPU story.

### Poseidon2-Style Merkle Forest

- Poseidon2-style hash over the Goldilocks field
- CPU reference hash, forest root builder, and proof path generation
- CUDA leaf and parent hash kernels
- intermediate Merkle levels kept GPU-resident
- host-transfer-included timing and CUDA-event device-only timing
- CPU verification of generated authentication paths

Full-mode benchmark forest shapes:

- one tree with `2^20` leaves
- `1024` trees with `2^10` leaves
- `65536` small authentication trees with `16` leaves each

## Why These Workloads Matter

[NVIDIA cuPQC](https://developer.nvidia.com/cupqc) directly targets ML-KEM and
ML-DSA and highlights SHA-2, SHA-3, SHAKE, Poseidon2, and Merkle trees. Its
public material reports H100 throughput for ML-KEM-768 operations in millions
of ops/sec and shows the expected CUDA pattern for Merkle workloads: larger
trees expose more parallelism.

[ZKProphet](https://arxiv.org/html/2509.22684v1) identifies MSM and NTT as the
dominant GPU prover kernels and calls out CPU-GPU transfer and kernel-launch
overheads as important bottlenecks. This repo emphasizes batching, residency,
NTTs, and Merkle forests for that reason.

The [Cambridge/OpenTitan NTT acceleration paper](https://www.cl.cam.ac.uk/~fms27/papers/2024-UrquhartStajano-acceleration.pdf)
frames NTT as a core bottleneck for Kyber/ML-KEM and Dilithium/ML-DSA style
systems.

## Build

Requirements:

- CMake 3.24+
- C++20 compiler
- NVIDIA CUDA Toolkit for GPU builds

CPU-only:

```bash
./scripts/build.sh -DCPB_ENABLE_CUDA=OFF
```

CUDA:

```bash
./scripts/build.sh -DCPB_ENABLE_CUDA=ON
```

Pin an architecture when building on a known GPU:

```bash
./scripts/build.sh -DCPB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
```

## Test

CPU-only CI path:

```bash
./scripts/test.sh -DCPB_ENABLE_CUDA=OFF
```

CUDA correctness path:

```bash
./scripts/test.sh -DCPB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
```

The correctness suite checks deterministic vectors, CPU roundtrips, edge-case
validation, and CPU-vs-CUDA parity when CUDA is available. Mismatch reports
include the first mismatching index, coefficient, byte, or digest word.

## Benchmark

Quick mode is the default:

```bash
./scripts/bench.sh -DCPB_ENABLE_CUDA=OFF
./scripts/bench.sh -DCPB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
```

Full CUDA mode runs the large batched ML-KEM and Poseidon2 forest rows:

```bash
CPB_BENCH_MODE=full ./scripts/bench.sh -DCPB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
```

Full scalar CPU Poseidon2 rows are opt-in because they are intentionally large:

```bash
CPB_POSEIDON2_FULL_CPU=1 CPB_BENCH_MODE=full ./scripts/bench.sh -DCPB_ENABLE_CUDA=OFF
```

Benchmark output prints the mode, GPU name, CUDA runtime/driver versions,
architecture, elapsed time, throughput, and whether Poseidon2 timings include
host transfers or device-only CUDA event timing.

Current public CUDA wrappers allocate and copy per call. Those numbers are
honest wrapper timings, not persistent-buffer optimized timings.

See [docs/benchmark-methodology.md](docs/benchmark-methodology.md) for the exact
recording checklist.

## CUDA Validation

Latest quick CUDA validation was run on Brev `cuda-pqc-t4-cheap`, machine type
`n1-highcpu-4:nvidia-tesla-t4:1`, listed by Brev at `$0.59/hr`.

- GPU: NVIDIA Tesla T4, `sm_75`, 14.56 GiB visible memory
- CUDA container: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- CUDA compiler: NVIDIA 12.4.131
- CMake: 3.29.6
- Command shape: `./scripts/test.sh -DCPB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`
- CUDA build passed
- correctness tests passed
- quick benchmark passed

Latest quick-mode T4 results:

| Primitive | Workload | Time ms | Throughput |
|-----------|----------|--------:|-----------:|
| Field mul | `n=65536` | 0.747 | 87.77 Mops/s |
| Radix-2 NTT | `n=16384` | 0.684 | 23.96 Mops/s |
| SHA-256 Merkle | `n=16384` | 0.882 | 18.57 Mops/s |
| ML-KEM NTT | `batch=1024` | 0.876 | 1168.626 Kpoly/s |
| ML-KEM poly mul | `batch=1024` | 1.584 | 646.563 Kpoly/s |

Latest quick-mode Poseidon2-style Merkle forest rows:

| Workload | Host ms | Device ms | Host Mhash/s | Host GB/s | Device Mhash/s | Device GB/s |
|----------|--------:|----------:|-------------:|----------:|---------------:|------------:|
| `1 x 2^10 leaves` | 4.163 | 3.919 | 0.49 | 0.03 | 0.52 | 0.03 |
| `32 x 2^10` | 6.518 | 5.659 | 10.05 | 0.60 | 11.58 | 0.69 |

The quick-mode T4 numbers include current wrapper allocation/copy overhead, and
the Poseidon2 rows also report device-only CUDA event timing. They are intended
as a cheap smoke benchmark, not the full throughput headline.

Previously recorded full-mode T4 reference on Brev `g4dn.xlarge`:

| Workload | Host ms | Device ms | Host Mhash/s | Host GB/s | Device Mhash/s | Device GB/s |
|----------|--------:|----------:|-------------:|----------:|---------------:|------------:|
| `1 x 2^20 leaves` | 46.208 | 31.920 | 45.39 | 2.71 | 65.70 | 3.92 |
| `1024 x 2^10` | 47.932 | 31.493 | 43.73 | 2.61 | 66.56 | 3.97 |
| `65536 x 16` | 44.943 | 29.841 | 45.20 | 2.69 | 68.08 | 4.06 |

Previously recorded full-mode batched ML-KEM/Kyber-style rows on T4:

| Primitive | Workload | Time ms | Throughput Kpoly/s |
|-----------|----------|--------:|-------------------:|
| ML-KEM NTT | `batch=1` | 0.244 | 4.094 |
| ML-KEM NTT | `batch=1024` | 0.786 | 1303.056 |
| ML-KEM NTT | `batch=10000` | 4.394 | 2276.023 |
| ML-KEM NTT | `batch=100000` | 59.778 | 1672.846 |
| ML-KEM poly mul | `batch=1` | 0.583 | 1.715 |
| ML-KEM poly mul | `batch=1024` | 1.198 | 854.695 |
| ML-KEM poly mul | `batch=10000` | 8.330 | 1200.443 |
| ML-KEM poly mul | `batch=100000` | 82.597 | 1210.699 |

Historical A6000 validation for the Goldilocks/SHA-256 path only:

| Primitive | Size | Baseline | Current | Change |
|-----------|-----:|---------:|--------:|-------:|
| NTT | 4096 | 4.32 Mops/s | 10.90 Mops/s | 2.52x |
| NTT | 16384 | 12.57 Mops/s | 31.25 Mops/s | 2.49x |
| Field mul | 65536 | 84.91 Mops/s | 94.82 Mops/s | 1.12x |
| Merkle SHA-256 | 16384 | 39.97 Mops/s | 41.70 Mops/s | 1.04x |

The A6000 table predates the ML-KEM and Poseidon2-style forest additions. Do not
compare it against the T4 tables as if they were from the same machine or code
path.

## Architecture Notes

- CPU references are deliberately direct and easy to audit.
- CUDA vector kernels use one thread per field element.
- CUDA NTTs use explicit stage kernels and device twiddle generation.
- Batched ML-KEM kernels map each polynomial to a CUDA block.
- Poseidon2-style Merkle forests copy leaves once, alternate resident level
  buffers, and copy only roots back.
- Goldilocks CUDA multiplication uses `__umul64hi` to recover the high half of
  a 64x64 product and folds with `2^64 = 2^32 - 1 mod p`.

## Repository Layout

```text
include/   public headers
src/       CPU and CUDA implementations
tests/     correctness tests
bench/     benchmark executable
scripts/   build, test, and benchmark helpers
docs/      design and benchmark notes
```

## Focused Future Work

- Standardized Poseidon2 parameters and published vectors.
- Persistent CUDA buffers for repeated benchmark runs.
- CUDA auth-path extraction from resident tree levels.
- Fused NTT stages and shared-memory tiling after benchmark baselines are stable.
- A reproducible benchmark matrix by GPU architecture.
