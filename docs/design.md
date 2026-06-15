# Architecture Notes

This repository intentionally starts with simple, inspectable kernels and CPU
reference code. The first goal is correctness and a clear benchmark harness,
not maximum throughput.

## Field

The finite field uses the Goldilocks prime:

```text
p = 2^64 - 2^32 + 1 = 18446744069414584321
```

This prime is common in proof-system prototypes because it fits in 64-bit
storage and has high two-adicity:

```text
p - 1 = 2^32 * (2^32 - 1)
```

The primitive root used by the implementation is `7`, which provides radix-2
roots of unity up to size `2^32`.

## CPU Reference Path

CPU field multiplication uses `unsigned __int128` for a direct modular
reduction. The CPU NTT is a conventional iterative Cooley-Tukey transform with
bit-reversal permutation.

## CUDA Path

The CUDA path mirrors the CPU path:

- vector add/mul uses one thread per field element,
- NTT uses a bit-reversal kernel followed by one kernel launch per stage,
- Merkle hashing uses one thread per leaf or parent hash.

The CUDA code deliberately favors readability over aggressive optimization.
CUDA field multiplication uses `__umul64hi` to recover the high half of the
64x64 product, then folds with the Goldilocks identity `2^64 = 2^32 - 1 mod p`.
The NTT precomputes stage twiddle powers once per stage on the device instead of
recomputing them for every butterfly. Future versions can replace repeated
kernel launches and host-managed Merkle levels with more optimized designs.

## Poseidon2 Merkle Forest

The `poseidon2_merkle_forest` path uses a Poseidon2-style Goldilocks permutation
with deterministic benchmark constants. It is intended for CUDA engineering
benchmarks, not production cryptography. Leaves contain eight Goldilocks field
elements, parent nodes absorb two four-word digests, and roots are four-word
digests.

The CUDA builder copies the input leaves once, hashes leaves into a device
buffer, alternates device-resident current/next level buffers until roots
remain, and copies only the final roots back. Benchmark output reports both
host-transfer-included wall time and device-only CUDA event time.

CPU proof-path generation builds a reference tree, emits sibling digests for
selected leaves, and verifies the path back to the deterministic root. This
keeps authentication logic auditable while the GPU path focuses on large forest
construction throughput.
