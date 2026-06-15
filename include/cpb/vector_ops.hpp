#pragma once

#include <cstdint>
#include <vector>

namespace cpb {

std::vector<std::uint64_t> vector_add_cpu(const std::vector<std::uint64_t>& a,
                                          const std::vector<std::uint64_t>& b);

std::vector<std::uint64_t> vector_mul_cpu(const std::vector<std::uint64_t>& a,
                                          const std::vector<std::uint64_t>& b);

}  // namespace cpb
