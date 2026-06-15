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
CUDA field multiplication currently uses a portable modular double-and-add
routine to avoid compiler-specific 128-bit device integer support. Future
versions can replace that, per-thread root exponentiation, repeated kernel
launches, and host-managed Merkle levels with more optimized designs.
