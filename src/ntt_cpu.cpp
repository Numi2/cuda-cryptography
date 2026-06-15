#include "cpb/ntt.hpp"

#include <algorithm>

#include "cpb/field.hpp"

namespace cpb {

namespace {

void bit_reverse(std::vector<std::uint64_t>& values) {
  const std::size_t n = values.size();
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1U;
    for (; (j & bit) != 0; bit >>= 1U) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(values[i], values[j]);
    }
  }
}

}  // namespace

void ntt_cpu_inplace(std::vector<std::uint64_t>& values, bool inverse) {
  const std::size_t n = values.size();
  field::log2_exact(n);
  bit_reverse(values);

  for (std::size_t len = 2; len <= n; len <<= 1U) {
    const std::uint64_t w_len = field::root_of_unity(len, inverse);
    for (std::size_t i = 0; i < n; i += len) {
      std::uint64_t w = 1;
      for (std::size_t j = 0; j < len / 2; ++j) {
        const std::uint64_t u = values[i + j];
        const std::uint64_t v = field::mul(values[i + j + len / 2], w);
        values[i + j] = field::add(u, v);
        values[i + j + len / 2] = field::sub(u, v);
        w = field::mul(w, w_len);
      }
    }
  }

  if (inverse) {
    const std::uint64_t inv_n = field::inverse(static_cast<std::uint64_t>(n));
    for (auto& value : values) {
      value = field::mul(value, inv_n);
    }
  }
}

std::vector<std::uint64_t> ntt_cpu(std::vector<std::uint64_t> values,
                                   bool inverse) {
  ntt_cpu_inplace(values, inverse);
  return values;
}

}  // namespace cpb
