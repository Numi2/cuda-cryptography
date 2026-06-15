#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace cpb::mlkem {

inline constexpr std::uint16_t kModulus = 3329;
inline constexpr std::size_t kPolyDegree = 256;
inline constexpr std::uint16_t kPrimitiveRoot256 = 17;

using Poly = std::array<std::uint16_t, kPolyDegree>;

std::uint16_t add_mod(std::uint16_t a, std::uint16_t b);
std::uint16_t sub_mod(std::uint16_t a, std::uint16_t b);
std::uint16_t mul_mod(std::uint16_t a, std::uint16_t b);
std::uint16_t pow_mod(std::uint16_t base, std::uint32_t exp);

void ntt_inplace(Poly& poly, bool inverse = false);
Poly ntt(Poly poly, bool inverse = false);

Poly poly_mul_schoolbook(const Poly& a, const Poly& b);
Poly poly_mul_ntt(const Poly& a, const Poly& b);
Poly deterministic_poly(std::uint32_t seed);

}  // namespace cpb::mlkem
