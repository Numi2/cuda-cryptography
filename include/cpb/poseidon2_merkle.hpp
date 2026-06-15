#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cpb::poseidon2 {

inline constexpr std::size_t kStateWidth = 12;
inline constexpr std::size_t kRateWords = 8;
inline constexpr std::size_t kDigestWords = 4;
inline constexpr std::size_t kFullRounds = 8;
inline constexpr std::size_t kPartialRounds = 22;

using Leaf = std::array<std::uint64_t, kRateWords>;
using Digest = std::array<std::uint64_t, kDigestWords>;

struct ForestShape {
  std::size_t tree_count;
  std::size_t leaves_per_tree;
};

struct AuthPath {
  std::size_t tree_index;
  std::size_t leaf_index;
  Leaf leaf;
  Digest root;
  std::vector<Digest> siblings;
};

struct ForestBuildResult {
  std::vector<Digest> roots;
  double host_ms = 0.0;
  double device_ms = 0.0;
  std::size_t hashes = 0;
  std::size_t bytes_absorbed = 0;
};

std::vector<Leaf> deterministic_leaves(std::size_t count, std::uint64_t seed);

Digest hash_leaf_cpu(const Leaf& leaf);
Digest hash_pair_cpu(const Digest& left, const Digest& right);

ForestBuildResult merkle_forest_roots_cpu(std::span<const Leaf> leaves,
                                          ForestShape shape);

AuthPath merkle_auth_path_cpu(std::span<const Leaf> leaves,
                              ForestShape shape,
                              std::size_t tree_index,
                              std::size_t leaf_index);

std::vector<AuthPath> merkle_auth_paths_cpu(std::span<const Leaf> leaves,
                                            ForestShape shape,
                                            std::span<const std::size_t> tree_indices,
                                            std::span<const std::size_t> leaf_indices);

bool verify_auth_path_cpu(const AuthPath& path);

std::size_t merkle_hash_count(ForestShape shape);
std::size_t absorbed_bytes_for_hashes(std::size_t hashes);

}  // namespace cpb::poseidon2
