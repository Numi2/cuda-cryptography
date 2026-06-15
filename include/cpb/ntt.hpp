#pragma once

#include <cstdint>
#include <vector>

namespace cpb {

void ntt_cpu_inplace(std::vector<std::uint64_t>& values, bool inverse = false);

std::vector<std::uint64_t> ntt_cpu(std::vector<std::uint64_t> values,
                                   bool inverse = false);

}  // namespace cpb
