# Poseidon2-Style Parameters

This project uses a Poseidon2-style permutation only for deterministic
benchmarking over the Goldilocks field. It is not a standardized Poseidon2
parameter set and must not be treated as production cryptography.

## Field

The field is Goldilocks:

```text
p = 2^64 - 2^32 + 1
```

This field is convenient for GPU benchmarks because multiplication can use the
identity:

```text
2^64 = 2^32 - 1 mod p
```

The CUDA path uses `__umul64hi` to recover the high half of a 64x64 product and
then folds the high words back into the field.

## State Shape

The benchmark permutation uses:

| Parameter | Value |
|-----------|------:|
| State width | 12 field words |
| Rate | 8 field words |
| Digest | 4 field words |
| Full rounds | 8 |
| Partial rounds | 22 |
| S-box | `x^7` |

The Merkle hash absorbs either one leaf or two child digests into the first rate
words, applies the permutation, and returns the first four state words.

## Domain Separation

The implementation uses a simple domain tag in state word `rate`:

| Hash input | Domain tag |
|------------|-----------:|
| Leaf, 8 field words | `1` |
| Parent, two 4-word digests | `2` |

These tags make benchmark leaf hashes and parent hashes distinct. They are not a
claim of standard Poseidon2 domain separation.

## Round Constants And MDS-Like Mixing

Round constants are generated deterministically from SplitMix64:

```text
seed = 0x706f736569646f6e ^ (round << 32) ^ lane
constant = splitmix64(seed) mod p
```

The mixing layer is intentionally compact:

```text
sum = state[0] + ... + state[11]
state[i] = diagonal(i) * state[i] + sum
diagonal(i) = 2 + ((i * 7) mod 13)
```

This is useful for repeatable CPU/CUDA benchmarking and parity tests, but it is
not a published Poseidon2 matrix.

## Test Vectors

The correctness suite pins deterministic vectors for:

- leaf hash
- pair hash
- small forest root
- CPU-vs-CUDA forest roots

Those vectors are implementation-regression vectors. They are not third-party
cryptographic test vectors.

## Production-Crypto Gap

To turn this into production-grade Poseidon2 work, the project would need:

- published parameter selection for the target field and security level
- published round constants and MDS matrix
- third-party test vectors
- constant-time review
- independent audit

Until then, this module should be read as GPU hash/Merkle engineering, not as a
secure Poseidon2 library.
