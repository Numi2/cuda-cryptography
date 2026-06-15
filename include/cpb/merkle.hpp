#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cpb/sha256.hpp"

namespace cpb {

Sha256Digest merkle_root_cpu(const std::vector<std::uint64_t>& leaves);

std::array<std::uint8_t, 8> field_element_to_le_bytes(std::uint64_t value);

}  // namespace cpb
