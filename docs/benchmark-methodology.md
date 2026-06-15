# Benchmark Methodology

Use this checklist when recording numbers for `cuda-cryptography`.

## Modes

Quick CPU-only smoke run:

```bash
./scripts/test.sh -DCPB_ENABLE_CUDA=OFF
./scripts/bench.sh -DCPB_ENABLE_CUDA=OFF
```

Quick CUDA run:

```bash
./scripts/test.sh -DCPB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<sm>
./scripts/bench.sh -DCPB_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<sm>
```

Full CUDA run:

```bash
CPB_BENCH_MODE=full ./scripts/bench.sh \
  -DCPB_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<sm>
```

Full scalar CPU Poseidon2-style forest rows are opt-in:

```bash
CPB_POSEIDON2_FULL_CPU=1 CPB_BENCH_MODE=full ./scripts/bench.sh \
  -DCPB_ENABLE_CUDA=OFF
```

## Machine Fields To Record

- Git commit hash.
- Full command line.
- Build type and `CMAKE_CUDA_ARCHITECTURES`.
- GPU model, SM architecture, and memory size.
- CUDA runtime and driver versions printed by `cpb_bench`.
- CPU model and core count when reporting CPU rows.
- Cloud instance type and whether the instance is shared or dedicated.
- Any unusual clock, power, MIG, or container constraints.

## Timing Semantics

Most public CUDA wrapper benchmark rows include allocation, host-to-device copy,
kernels, synchronization, and device-to-host copy. This is intentional: the
numbers are honest for the current public API.

Resident-buffer rows are separate. They allocate and copy inputs once, reuse GPU
scratch buffers across repeated timed iterations, and report CUDA-event
device-only timing. ML-KEM resident rows include device-to-device work-buffer
resets so each iteration starts from the same input. Poseidon2 resident rows
rebuild the tree from resident leaves and copy roots back only after timing.

Poseidon2-style Merkle forest rows report two timings:

- `Host ms`: allocation, host-to-device leaf copy, kernels, synchronization, and
  root copy back.
- `Device ms`: CUDA-event timing for leaf and parent kernels while leaves and
  intermediate levels are GPU-resident.

Do not label wrapper rows as resident-buffer performance. Do not compare wrapper
rows and resident rows without explaining the timing semantics.

## Interpreting Results

- ML-KEM/Kyber-style `n=256` work is too small for a single polynomial to be a
  useful GPU headline. Use batched rows.
- Poseidon2-style Merkle forests should improve as total leaves increase because
  larger levels expose more parallelism.
- CPU reference rows are correctness and baseline rows, not hand-vectorized CPU
  cryptography.
- Compare numbers only across the same GPU model, driver/runtime, build flags,
  and benchmark mode.

## Cloud Cost Hygiene

Prefer stoppable low-cost CUDA instances for validation. Stop the instance after
the run instead of deleting it when possible, and record the instance type in the
benchmark notes.
