#pragma once

#include <cstdint>
#include <vector>

#include "cpb/mlkem.hpp"
#include "cpb/sha256.hpp"

namespace cpb::cuda {

bool is_available();

std::vector<std::uint64_t> vector_add(const std::vector<std::uint64_t>& a,
                                      const std::vector<std::uint64_t>& b);

std::vector<std::uint64_t> vector_mul(const std::vector<std::uint64_t>& a,
                                      const std::vector<std::uint64_t>& b);

std::vector<std::uint64_t> ntt(const std::vector<std::uint64_t>& values,
                               bool inverse = false);

Sha256Digest merkle_root(const std::vector<std::uint64_t>& leaves);

mlkem::Poly mlkem_poly_mul_ntt(const mlkem::Poly& a, const mlkem::Poly& b);

std::vector<mlkem::Poly> mlkem_ntt_batch(const std::vector<mlkem::Poly>& polys,
                                         bool inverse = false);

std::vector<mlkem::Poly> mlkem_poly_mul_ntt_batch(
    const std::vector<mlkem::Poly>& a,
    const std::vector<mlkem::Poly>& b);

}  // namespace cpb::cuda
