#include "cpb/cuda_primitives.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "cpb/field.hpp"

namespace cpb::cuda {

namespace {

constexpr std::uint64_t kModulus = field::kModulus;

void check(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
  }
}

__device__ __forceinline__ std::uint64_t d_add(std::uint64_t a, std::uint64_t b) {
  const std::uint64_t sum = a + b;
  if (sum < a || sum >= kModulus) {
    return sum - kModulus;
  }
  return sum;
}

__device__ __forceinline__ std::uint64_t d_sub(std::uint64_t a, std::uint64_t b) {
  return (a >= b) ? (a - b) : (kModulus - (b - a));
}

__device__ __forceinline__ void goldilocks_fold_once(std::uint64_t& lo,
                                                     std::uint64_t& hi) {
  std::uint64_t term_lo = hi << 32U;
  std::uint64_t term_hi = hi >> 32U;
  const std::uint64_t borrow = term_lo < hi ? 1ULL : 0ULL;
  term_lo -= hi;
  term_hi -= borrow;

  const std::uint64_t sum = lo + term_lo;
  const std::uint64_t carry = sum < lo ? 1ULL : 0ULL;
  lo = sum;
  hi = term_hi + carry;
}

__device__ __forceinline__ std::uint64_t d_mul(std::uint64_t a, std::uint64_t b) {
  std::uint64_t lo = a * b;
  std::uint64_t hi = __umul64hi(a, b);

  // For p = 2^64 - 2^32 + 1, 2^64 == 2^32 - 1 mod p.
  goldilocks_fold_once(lo, hi);
  goldilocks_fold_once(lo, hi);
  goldilocks_fold_once(lo, hi);

  if (lo >= kModulus) {
    lo -= kModulus;
  }
  if (lo >= kModulus) {
    lo -= kModulus;
  }
  return lo;
}

__device__ std::uint64_t d_pow(std::uint64_t base, std::uint64_t exp) {
  std::uint64_t result = 1;
  while (exp != 0) {
    if ((exp & 1U) != 0) {
      result = d_mul(result, base);
    }
    base = d_mul(base, base);
    exp >>= 1U;
  }
  return result;
}

__global__ void vector_add_kernel(const std::uint64_t* a,
                                  const std::uint64_t* b,
                                  std::uint64_t* out,
                                  std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = d_add(a[i], b[i]);
  }
}

__global__ void vector_mul_kernel(const std::uint64_t* a,
                                  const std::uint64_t* b,
                                  std::uint64_t* out,
                                  std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = d_mul(a[i], b[i]);
  }
}

__device__ std::size_t reverse_bits(std::size_t value, int bits) {
  return __brevll(static_cast<unsigned long long>(value)) >> (64 - bits);
}

__global__ void bit_reverse_kernel(std::uint64_t* values, std::size_t n, int bits) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const std::size_t j = reverse_bits(i, bits);
  if (i < j) {
    const std::uint64_t tmp = values[i];
    values[i] = values[j];
    values[j] = tmp;
  }
}

__global__ void ntt_stage_kernel(std::uint64_t* values,
                                 std::size_t n,
                                 std::size_t len,
                                 const std::uint64_t* twiddles) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t butterflies = n / 2;
  if (tid >= butterflies) {
    return;
  }

  const std::size_t half = len / 2;
  const std::size_t j = tid & (half - 1);
  const std::size_t i = ((tid - j) << 1U) + j;

  const std::uint64_t w = twiddles[j];
  const std::uint64_t u = values[i];
  const std::uint64_t v = d_mul(values[i + half], w);
  values[i] = d_add(u, v);
  values[i + half] = d_sub(u, v);
}

__global__ void twiddle_kernel(std::uint64_t* twiddles,
                               std::size_t half,
                               std::uint64_t w_len) {
  const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < half) {
    twiddles[j] = d_pow(w_len, j);
  }
}

__global__ void scale_kernel(std::uint64_t* values, std::size_t n, std::uint64_t scalar) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    values[i] = d_mul(values[i], scalar);
  }
}

__device__ __constant__ std::uint32_t kSha256K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

__device__ std::uint32_t rotr32(std::uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32U - bits));
}

__device__ std::uint32_t load_be32(const std::uint8_t* p) {
  return (static_cast<std::uint32_t>(p[0]) << 24U) |
         (static_cast<std::uint32_t>(p[1]) << 16U) |
         (static_cast<std::uint32_t>(p[2]) << 8U) |
         static_cast<std::uint32_t>(p[3]);
}

__device__ void store_be32(std::uint32_t value, std::uint8_t* out) {
  out[0] = static_cast<std::uint8_t>(value >> 24U);
  out[1] = static_cast<std::uint8_t>(value >> 16U);
  out[2] = static_cast<std::uint8_t>(value >> 8U);
  out[3] = static_cast<std::uint8_t>(value);
}

__device__ void sha256_compress(const std::uint8_t block[64], std::uint32_t state[8]) {
  std::uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = load_be32(block + i * 4);
  }
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^
                             (w[i - 15] >> 3U);
    const std::uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^
                             (w[i - 2] >> 10U);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }

  std::uint32_t a = state[0];
  std::uint32_t b = state[1];
  std::uint32_t c = state[2];
  std::uint32_t d = state[3];
  std::uint32_t e = state[4];
  std::uint32_t f = state[5];
  std::uint32_t g = state[6];
  std::uint32_t h = state[7];

  for (int i = 0; i < 64; ++i) {
    const std::uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
    const std::uint32_t ch = (e & f) ^ ((~e) & g);
    const std::uint32_t temp1 = h + s1 + ch + kSha256K[i] + w[i];
    const std::uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

__device__ void sha256_device(const std::uint8_t* input,
                              std::size_t len,
                              std::uint8_t digest[32]) {
  std::uint32_t state[8] = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};

  std::size_t offset = 0;
  while (len - offset >= 64) {
    std::uint8_t block[64];
    for (int i = 0; i < 64; ++i) {
      block[i] = input[offset + i];
    }
    sha256_compress(block, state);
    offset += 64;
  }

  std::uint8_t block[64] = {};
  const std::size_t remaining = len - offset;
  for (std::size_t i = 0; i < remaining; ++i) {
    block[i] = input[offset + i];
  }
  block[remaining] = 0x80;

  if (remaining >= 56) {
    sha256_compress(block, state);
    for (auto& byte : block) {
      byte = 0;
    }
  }

  const std::uint64_t bit_len = static_cast<std::uint64_t>(len) * 8U;
  for (int i = 0; i < 8; ++i) {
    block[63 - i] = static_cast<std::uint8_t>(bit_len >> (8U * i));
  }
  sha256_compress(block, state);

  for (int i = 0; i < 8; ++i) {
    store_be32(state[i], digest + i * 4);
  }
}

__global__ void merkle_leaf_kernel(const std::uint64_t* leaves,
                                   std::uint8_t* hashes,
                                   std::size_t n) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  std::uint8_t bytes[8];
  const std::uint64_t value = leaves[i];
  for (int j = 0; j < 8; ++j) {
    bytes[j] = static_cast<std::uint8_t>(value >> (8U * j));
  }
  sha256_device(bytes, 8, hashes + i * 32);
}

__global__ void merkle_parent_kernel(const std::uint8_t* current,
                                     std::uint8_t* next,
                                     std::size_t current_count,
                                     std::size_t next_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= next_count) {
    return;
  }
  std::uint8_t bytes[64];
  const std::size_t left = 2 * i;
  const std::size_t right = (left + 1 < current_count) ? (left + 1) : left;
  for (int j = 0; j < 32; ++j) {
    bytes[j] = current[left * 32 + j];
    bytes[32 + j] = current[right * 32 + j];
  }
  sha256_device(bytes, 64, next + i * 32);
}

}  // namespace

bool is_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<std::uint64_t> vector_add(const std::vector<std::uint64_t>& a,
                                      const std::vector<std::uint64_t>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("cuda::vector_add input sizes differ");
  }
  if (a.empty()) {
    return {};
  }

  std::uint64_t* d_a = nullptr;
  std::uint64_t* d_b = nullptr;
  std::uint64_t* d_out = nullptr;
  const std::size_t bytes = a.size() * sizeof(std::uint64_t);
  check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
  check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
  check(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
  check(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy a to device");
  check(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy b to device");

  constexpr int kBlockSize = 256;
  const int blocks = static_cast<int>((a.size() + kBlockSize - 1) / kBlockSize);
  vector_add_kernel<<<blocks, kBlockSize>>>(d_a, d_b, d_out, a.size());
  check(cudaGetLastError(), "vector_add_kernel");
  check(cudaDeviceSynchronize(), "vector_add synchronize");

  std::vector<std::uint64_t> out(a.size());
  check(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost), "copy add output");
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  return out;
}

std::vector<std::uint64_t> vector_mul(const std::vector<std::uint64_t>& a,
                                      const std::vector<std::uint64_t>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("cuda::vector_mul input sizes differ");
  }
  if (a.empty()) {
    return {};
  }

  std::uint64_t* d_a = nullptr;
  std::uint64_t* d_b = nullptr;
  std::uint64_t* d_out = nullptr;
  const std::size_t bytes = a.size() * sizeof(std::uint64_t);
  check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
  check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
  check(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
  check(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy a to device");
  check(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy b to device");

  constexpr int kBlockSize = 256;
  const int blocks = static_cast<int>((a.size() + kBlockSize - 1) / kBlockSize);
  vector_mul_kernel<<<blocks, kBlockSize>>>(d_a, d_b, d_out, a.size());
  check(cudaGetLastError(), "vector_mul_kernel");
  check(cudaDeviceSynchronize(), "vector_mul synchronize");

  std::vector<std::uint64_t> out(a.size());
  check(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost), "copy mul output");
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  return out;
}

std::vector<std::uint64_t> ntt(const std::vector<std::uint64_t>& values, bool inverse) {
  const std::size_t n = values.size();
  const std::size_t bits = field::log2_exact(n);
  if (n == 1) {
    return values;
  }

  std::uint64_t* d_values = nullptr;
  std::uint64_t* d_twiddles = nullptr;
  const std::size_t bytes = n * sizeof(std::uint64_t);
  check(cudaMalloc(&d_values, bytes), "cudaMalloc ntt values");
  check(cudaMalloc(&d_twiddles, (n / 2) * sizeof(std::uint64_t)),
        "cudaMalloc ntt twiddles");
  check(cudaMemcpy(d_values, values.data(), bytes, cudaMemcpyHostToDevice),
        "copy ntt input to device");

  constexpr int kBlockSize = 256;
  int blocks = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  bit_reverse_kernel<<<blocks, kBlockSize>>>(d_values, n, static_cast<int>(bits));
  check(cudaGetLastError(), "bit_reverse_kernel");
  check(cudaDeviceSynchronize(), "bit_reverse synchronize");

  for (std::size_t len = 2; len <= n; len <<= 1U) {
    const std::uint64_t w_len = field::root_of_unity(len, inverse);
    const std::size_t half = len / 2;
    blocks = static_cast<int>((half + kBlockSize - 1) / kBlockSize);
    twiddle_kernel<<<blocks, kBlockSize>>>(d_twiddles, half, w_len);
    check(cudaGetLastError(), "twiddle_kernel");
    check(cudaDeviceSynchronize(), "twiddle synchronize");

    blocks = static_cast<int>(((n / 2) + kBlockSize - 1) / kBlockSize);
    ntt_stage_kernel<<<blocks, kBlockSize>>>(d_values, n, len, d_twiddles);
    check(cudaGetLastError(), "ntt_stage_kernel");
    check(cudaDeviceSynchronize(), "ntt stage synchronize");
  }

  if (inverse) {
    const std::uint64_t inv_n = field::inverse(static_cast<std::uint64_t>(n));
    blocks = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    scale_kernel<<<blocks, kBlockSize>>>(d_values, n, inv_n);
    check(cudaGetLastError(), "scale_kernel");
    check(cudaDeviceSynchronize(), "scale synchronize");
  }

  std::vector<std::uint64_t> out(n);
  check(cudaMemcpy(out.data(), d_values, bytes, cudaMemcpyDeviceToHost), "copy ntt output");
  cudaFree(d_values);
  cudaFree(d_twiddles);
  return out;
}

Sha256Digest merkle_root(const std::vector<std::uint64_t>& leaves) {
  if (leaves.empty()) {
    throw std::invalid_argument("Merkle tree requires at least one leaf");
  }

  const std::size_t leaf_bytes = leaves.size() * sizeof(std::uint64_t);
  std::uint64_t* d_leaves = nullptr;
  std::uint8_t* d_current = nullptr;
  std::uint8_t* d_next = nullptr;
  check(cudaMalloc(&d_leaves, leaf_bytes), "cudaMalloc merkle leaves");
  check(cudaMemcpy(d_leaves, leaves.data(), leaf_bytes, cudaMemcpyHostToDevice),
        "copy merkle leaves");
  check(cudaMalloc(&d_current, leaves.size() * 32), "cudaMalloc merkle current");
  check(cudaMalloc(&d_next, ((leaves.size() + 1) / 2) * 32), "cudaMalloc merkle next");

  constexpr int kBlockSize = 256;
  int blocks = static_cast<int>((leaves.size() + kBlockSize - 1) / kBlockSize);
  merkle_leaf_kernel<<<blocks, kBlockSize>>>(d_leaves, d_current, leaves.size());
  check(cudaGetLastError(), "merkle_leaf_kernel");
  check(cudaDeviceSynchronize(), "merkle leaf synchronize");

  std::size_t current_count = leaves.size();
  std::size_t next_capacity = (leaves.size() + 1) / 2;
  while (current_count > 1) {
    const std::size_t next_count = (current_count + 1) / 2;
    if (next_count > next_capacity) {
      cudaFree(d_next);
      check(cudaMalloc(&d_next, next_count * 32), "resize merkle next");
      next_capacity = next_count;
    }
    blocks = static_cast<int>((next_count + kBlockSize - 1) / kBlockSize);
    merkle_parent_kernel<<<blocks, kBlockSize>>>(d_current, d_next, current_count, next_count);
    check(cudaGetLastError(), "merkle_parent_kernel");
    check(cudaDeviceSynchronize(), "merkle parent synchronize");
    std::swap(d_current, d_next);
    current_count = next_count;
  }

  Sha256Digest root{};
  check(cudaMemcpy(root.data(), d_current, root.size(), cudaMemcpyDeviceToHost),
        "copy merkle root");
  cudaFree(d_leaves);
  cudaFree(d_current);
  cudaFree(d_next);
  return root;
}

}  // namespace cpb::cuda
