# cuda-cryptography

CUDA/C++20 benchmark and correctness suite for post-quantum and
proof-system-adjacent cryptographic kernels.

This repository is designed to show credible GPU cryptography engineering:
CPU reference implementations, CUDA kernels, deterministic correctness tests,
CMake, CI, benchmark output, and clear architecture notes. It is intentionally
small enough to read.

## Security Notice

This is research and benchmark code, not production cryptography.

The implementations are not audited, not constant-time, and not suitable for
protecting real assets. The Poseidon2 path uses a Poseidon2-style Goldilocks
permutation with deterministic benchmark constants; it is not a standardized
Poseidon2 parameter set.

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
- benchmark batches: `1`, `1024`, `10000`, `100000`

A single Kyber-sized polynomial is too small to make CUDA look good. The
headline is batched ML-KEM-style NTT/poly-mul.

### Poseidon2 Merkle Forest

- Poseidon2-style hash over the Goldilocks field
- CPU reference hash, forest root builder, and proof path generation
- CUDA leaf and parent hash kernels
- intermediate Merkle levels kept GPU-resident
- host-transfer-included timing
- device-only CUDA event timing
- throughput in hashes/sec and GB/s
- CPU verification of generated authentication paths

Benchmark forest shapes:

- one tree with `2^20` leaves
- `1024` trees with `2^10` leaves
- `65536` small authentication trees with `16` leaves each

Full-size CPU Poseidon2 forest rows are opt-in because they are intentionally
large scalar reference runs.

## Why These Workloads Matter

### NVIDIA cuPQC Alignment

[NVIDIA cuPQC](https://developer.nvidia.com/cupqc) directly targets ML-KEM and
ML-DSA, and its public material reports H100 throughput for ML-KEM-768
keygen/encap/decap in millions of operations per second.

cuPQC also highlights SHA-2, SHA-3, SHAKE, Poseidon2, and Merkle trees. Its
Merkle examples improve with larger trees, which is exactly the CUDA story:
more leaves expose more parallelism.

### ZK and Proof-System Kernels

[ZKProphet](https://arxiv.org/html/2509.22684v1) identifies MSM and NTT as the
dominant GPU prover kernels and calls out CPU-GPU transfer and kernel-launch
overheads as important bottlenecks. This repo therefore emphasizes batching,
resident device buffers, NTTs, and Merkle forests instead of isolated tiny
operations.

### Lattice/FHE/PQC NTTs

The [Cambridge/OpenTitan NTT acceleration paper](https://www.cl.cam.ac.uk/~fms27/papers/2024-UrquhartStajano-acceleration.pdf)
frames NTT as a core bottleneck for Kyber/ML-KEM and Dilithium/ML-DSA style
systems. This repo uses the same idea at CUDA benchmark scale.

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
./scripts/build.sh
```

Use `CMAKE_CUDA_ARCHITECTURES` if you want to pin an architecture:

```bash
./scripts/build.sh -DCMAKE_CUDA_ARCHITECTURES=75
```

## Test

```bash
./scripts/test.sh
```

The correctness suite checks:

- deterministic field vectors
- SHA-256 vector for `abc`
- CPU NTT roundtrip
- CPU-vs-CUDA field/vector/NTT/Merkle outputs when CUDA is available
- ML-KEM deterministic coefficient/product vectors
- ML-KEM CPU NTT multiplication against schoolbook multiplication
- CUDA batched ML-KEM NTT/poly-mul against CPU reference
- Poseidon2-style deterministic Merkle roots
- Poseidon2-style auth path generation and CPU verification
- CUDA Poseidon2 Merkle forest roots against CPU reference

Mismatch reports include the first mismatching index, coefficient, byte, or
digest word.

## Benchmark

```bash
./scripts/bench.sh
```

Run full scalar CPU Poseidon2 forest rows:

```bash
CPB_POSEIDON2_FULL_CPU=1 ./scripts/bench.sh -DCPB_ENABLE_CUDA=OFF
```

Benchmark output includes:

- elapsed time
- field/vector/NTT throughput
- ML-KEM polynomial throughput
- Poseidon2 forest host-transfer-included timing
- Poseidon2 forest device-only timing
- hashes/sec and GB/s

## Latest CUDA Validation

Validated on Brev `g4dn.xlarge` with NVIDIA Tesla T4, CUDA 12.4 container,
`CMAKE_CUDA_ARCHITECTURES=75`.

- CUDA build passed
- correctness tests passed
- benchmark ran end-to-end

Poseidon2-style Merkle forest results on T4:

| Workload | Host ms | Device ms | Host Mhash/s | Host GB/s | Device Mhash/s | Device GB/s |
|----------|--------:|----------:|-------------:|----------:|---------------:|------------:|
| `1 x 2^20 leaves` | 46.208 | 31.920 | 45.39 | 2.71 | 65.70 | 3.92 |
| `1024 x 2^10` | 47.932 | 31.493 | 43.73 | 2.61 | 66.56 | 3.97 |
| `65536 x 16` | 44.943 | 29.841 | 45.20 | 2.69 | 68.08 | 4.06 |

Batched ML-KEM/Kyber-style results on T4:

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

Historical A6000 validation for the Goldilocks/SHA-256 path:

| Primitive | Size | Baseline | Current | Change |
|-----------|-----:|---------:|--------:|-------:|
| NTT | 4096 | 4.32 Mops/s | 10.90 Mops/s | 2.52x |
| NTT | 16384 | 12.57 Mops/s | 31.25 Mops/s | 2.49x |
| Field mul | 65536 | 84.91 Mops/s | 94.82 Mops/s | 1.12x |
| Merkle SHA-256 | 16384 | 39.97 Mops/s | 41.70 Mops/s | 1.04x |

Actual numbers depend on GPU model, CUDA version, clocks, transfer overhead,
and benchmark shape.

## Architecture Notes

The code favors clarity first:

- CPU references are straightforward and intentionally boring.
- CUDA vector kernels use one thread per field element.
- CUDA NTTs use explicit stage kernels and device twiddle generation.
- Batched ML-KEM kernels map each polynomial to a CUDA block.
- Poseidon2 Merkle forests copy leaves once, alternate resident level buffers,
  and copy only roots back.
- Poseidon2 benchmarks report both host wall time and device event time.

Goldilocks CUDA multiplication uses `__umul64hi` to recover the high half of a
64x64 product and folds with `2^64 = 2^32 - 1 mod p`.

## Repository Layout

```text
include/   public headers
src/       CPU and CUDA implementations
tests/     correctness tests
bench/     benchmark executable
scripts/   build, test, and benchmark helpers
docs/      design notes
```

## Future Work

- standardized Poseidon2 parameters and published test vectors
- CUDA auth-path extraction directly from resident tree levels
- fused NTT stages and shared-memory tiling
- persistent CUDA buffers for repeated benchmark runs
- larger benchmark matrix by GPU architecture
- comparison against cuPQC or other established libraries where licensing allows
