#include "cpb/merkle.hpp"

#include <stdexcept>

namespace cpb {

std::array<std::uint8_t, 8> field_element_to_le_bytes(std::uint64_t value) {
  std::array<std::uint8_t, 8> out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint8_t>(value >> (8U * i));
  }
  return out;
}

Sha256Digest merkle_root_cpu(const std::vector<std::uint64_t>& leaves) {
  if (leaves.empty()) {
    throw std::invalid_argument("Merkle tree requires at least one leaf");
  }

  std::vector<Sha256Digest> level;
  level.reserve(leaves.size());
  for (const std::uint64_t leaf : leaves) {
    const auto bytes = field_element_to_le_bytes(leaf);
    level.push_back(sha256(bytes));
  }

  while (level.size() > 1) {
    std::vector<Sha256Digest> next((level.size() + 1) / 2);
    for (std::size_t i = 0; i < next.size(); ++i) {
      const Sha256Digest& left = level[2 * i];
      const Sha256Digest& right =
          (2 * i + 1 < level.size()) ? level[2 * i + 1] : level[2 * i];
      std::array<std::uint8_t, 64> bytes{};
      std::copy(left.begin(), left.end(), bytes.begin());
      std::copy(right.begin(), right.end(), bytes.begin() + 32);
      next[i] = sha256(bytes);
    }
    level = std::move(next);
  }

  return level.front();
}

}  // namespace cpb
