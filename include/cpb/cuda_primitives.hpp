#pragma once

#include <cstdint>
#include <vector>

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

}  // namespace cpb::cuda
