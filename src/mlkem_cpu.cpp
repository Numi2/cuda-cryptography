#include "cpb/mlkem.hpp"

#include <algorithm>

namespace cpb::mlkem {

namespace {

void bit_reverse(Poly& poly) {
  for (std::size_t i = 1, j = 0; i < kPolyDegree; ++i) {
    std::size_t bit = kPolyDegree >> 1U;
    for (; (j & bit) != 0; bit >>= 1U) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(poly[i], poly[j]);
    }
  }
}

}  // namespace

std::uint16_t add_mod(std::uint16_t a, std::uint16_t b) {
  const std::uint32_t sum = static_cast<std::uint32_t>(a) + b;
  return static_cast<std::uint16_t>(sum >= kModulus ? sum - kModulus : sum);
}

std::uint16_t sub_mod(std::uint16_t a, std::uint16_t b) {
  return static_cast<std::uint16_t>(a >= b ? a - b : kModulus + a - b);
}

std::uint16_t mul_mod(std::uint16_t a, std::uint16_t b) {
  return static_cast<std::uint16_t>((static_cast<std::uint32_t>(a) * b) % kModulus);
}

std::uint16_t pow_mod(std::uint16_t base, std::uint32_t exp) {
  std::uint16_t result = 1;
  while (exp != 0) {
    if ((exp & 1U) != 0) {
      result = mul_mod(result, base);
    }
    base = mul_mod(base, base);
    exp >>= 1U;
  }
  return result;
}

void ntt_inplace(Poly& poly, bool inverse) {
  bit_reverse(poly);

  for (std::size_t len = 2; len <= kPolyDegree; len <<= 1U) {
    std::uint16_t w_len =
        pow_mod(kPrimitiveRoot256, static_cast<std::uint32_t>(kPolyDegree / len));
    if (inverse) {
      w_len = pow_mod(w_len, kModulus - 2);
    }

    for (std::size_t i = 0; i < kPolyDegree; i += len) {
      std::uint16_t w = 1;
      for (std::size_t j = 0; j < len / 2; ++j) {
        const std::uint16_t u = poly[i + j];
        const std::uint16_t v = mul_mod(poly[i + j + len / 2], w);
        poly[i + j] = add_mod(u, v);
        poly[i + j + len / 2] = sub_mod(u, v);
        w = mul_mod(w, w_len);
      }
    }
  }

  if (inverse) {
    const std::uint16_t inv_n = pow_mod(static_cast<std::uint16_t>(kPolyDegree),
                                        kModulus - 2);
    for (auto& coeff : poly) {
      coeff = mul_mod(coeff, inv_n);
    }
  }
}

Poly ntt(Poly poly, bool inverse) {
  ntt_inplace(poly, inverse);
  return poly;
}

Poly poly_mul_schoolbook(const Poly& a, const Poly& b) {
  Poly out{};
  for (std::size_t i = 0; i < kPolyDegree; ++i) {
    for (std::size_t j = 0; j < kPolyDegree; ++j) {
      const std::size_t k = (i + j) & (kPolyDegree - 1);
      out[k] = add_mod(out[k], mul_mod(a[i], b[j]));
    }
  }
  return out;
}

Poly poly_mul_ntt(const Poly& a, const Poly& b) {
  Poly a_eval = ntt(a);
  Poly b_eval = ntt(b);
  for (std::size_t i = 0; i < kPolyDegree; ++i) {
    a_eval[i] = mul_mod(a_eval[i], b_eval[i]);
  }
  ntt_inplace(a_eval, true);
  return a_eval;
}

Poly deterministic_poly(std::uint32_t seed) {
  Poly poly{};
  std::uint32_t x = seed;
  for (auto& coeff : poly) {
    x = x * 1664525U + 1013904223U;
    coeff = static_cast<std::uint16_t>(x % kModulus);
  }
  return poly;
}

}  // namespace cpb::mlkem
