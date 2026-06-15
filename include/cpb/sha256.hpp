#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace cpb {

using Sha256Digest = std::array<std::uint8_t, 32>;

Sha256Digest sha256(std::span<const std::uint8_t> bytes);

}  // namespace cpb
