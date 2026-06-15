#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cpb::field {

// Goldilocks prime: 2^64 - 2^32 + 1.
// p - 1 = 2^32 * (2^32 - 1), so radix-2 NTTs up to size 2^32 are available.
inline constexpr std::uint64_t kModulus = 18446744069414584321ULL;
inline constexpr std::uint64_t kPrimitiveRoot = 7ULL;
inline constexpr std::size_t kMaxPowerOfTwoNtt = 32;

inline std::uint64_t add(std::uint64_t a, std::uint64_t b) {
  const std::uint64_t sum = a + b;
  if (sum < a || sum >= kModulus) {
    return sum - kModulus;
  }
  return sum;
}

inline std::uint64_t sub(std::uint64_t a, std::uint64_t b) {
  if (a >= b) {
    return a - b;
  }
  return kModulus - (b - a);
}

inline std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
  const auto product = static_cast<unsigned __int128>(a) * b;
  return static_cast<std::uint64_t>(product % kModulus);
}

inline std::uint64_t pow(std::uint64_t base, std::uint64_t exp) {
  std::uint64_t result = 1;
  while (exp != 0) {
    if ((exp & 1U) != 0) {
      result = mul(result, base);
    }
    base = mul(base, base);
    exp >>= 1U;
  }
  return result;
}

inline std::uint64_t inverse(std::uint64_t value) {
  if (value == 0) {
    throw std::invalid_argument("zero has no field inverse");
  }
  return pow(value, kModulus - 2);
}

inline bool is_power_of_two(std::size_t n) {
  return n != 0 && (n & (n - 1)) == 0;
}

inline std::size_t log2_exact(std::size_t n) {
  if (!is_power_of_two(n)) {
    throw std::invalid_argument("size must be a power of two");
  }
  std::size_t bits = 0;
  while (n > 1) {
    n >>= 1U;
    ++bits;
  }
  return bits;
}

inline std::uint64_t root_of_unity(std::size_t n, bool inverse_root = false) {
  const std::size_t bits = log2_exact(n);
  if (bits > kMaxPowerOfTwoNtt) {
    throw std::invalid_argument("NTT size exceeds field two-adicity");
  }
  std::uint64_t root = pow(kPrimitiveRoot, (kModulus - 1) / n);
  if (inverse_root) {
    root = inverse(root);
  }
  return root;
}

}  // namespace cpb::field
