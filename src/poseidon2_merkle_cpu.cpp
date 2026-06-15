#include "cpb/poseidon2_merkle.hpp"

#include <chrono>
#include <stdexcept>

#include "cpb/field.hpp"

namespace cpb::poseidon2 {

namespace {

using State = std::array<std::uint64_t, kStateWidth>;

bool is_power_of_two(std::size_t n) {
  return n != 0 && (n & (n - 1)) == 0;
}

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31U);
}

std::uint64_t round_constant(std::size_t round, std::size_t lane) {
  const std::uint64_t seed = 0x706f736569646f6eULL ^
                             (static_cast<std::uint64_t>(round) << 32U) ^
                             static_cast<std::uint64_t>(lane);
  return splitmix64(seed) % field::kModulus;
}

std::uint64_t diagonal(std::size_t lane) {
  return 2 + static_cast<std::uint64_t>((lane * 7) % 13);
}

std::uint64_t sbox(std::uint64_t x) {
  const std::uint64_t x2 = field::mul(x, x);
  const std::uint64_t x4 = field::mul(x2, x2);
  return field::mul(field::mul(x4, x2), x);
}

void mix(State& state) {
  std::uint64_t sum = 0;
  for (const auto value : state) {
    sum = field::add(sum, value);
  }
  for (std::size_t i = 0; i < state.size(); ++i) {
    state[i] = field::add(field::mul(state[i], diagonal(i)), sum);
  }
}

void add_round_constants(State& state, std::size_t round) {
  for (std::size_t i = 0; i < state.size(); ++i) {
    state[i] = field::add(state[i], round_constant(round, i));
  }
}

void poseidon2_permute(State& state) {
  mix(state);
  std::size_t round = 0;
  for (; round < kFullRounds / 2; ++round) {
    add_round_constants(state, round);
    for (auto& value : state) {
      value = sbox(value);
    }
    mix(state);
  }
  for (; round < kFullRounds / 2 + kPartialRounds; ++round) {
    add_round_constants(state, round);
    state[0] = sbox(state[0]);
    mix(state);
  }
  for (; round < kFullRounds + kPartialRounds; ++round) {
    add_round_constants(state, round);
    for (auto& value : state) {
      value = sbox(value);
    }
    mix(state);
  }
}

Digest digest_from_state(const State& state) {
  Digest digest{};
  for (std::size_t i = 0; i < digest.size(); ++i) {
    digest[i] = state[i];
  }
  return digest;
}

void validate_shape(std::span<const Leaf> leaves, ForestShape shape) {
  if (shape.tree_count == 0 || !is_power_of_two(shape.leaves_per_tree)) {
    throw std::invalid_argument("Merkle forest shape must use nonzero power-of-two trees");
  }
  if (leaves.size() != shape.tree_count * shape.leaves_per_tree) {
    throw std::invalid_argument("leaf count does not match Merkle forest shape");
  }
}

std::vector<std::vector<Digest>> build_tree_levels(std::span<const Leaf> tree_leaves) {
  std::vector<std::vector<Digest>> levels;
  levels.emplace_back(tree_leaves.size());
  for (std::size_t i = 0; i < tree_leaves.size(); ++i) {
    levels[0][i] = hash_leaf_cpu(tree_leaves[i]);
  }

  while (levels.back().size() > 1) {
    const std::size_t current_level = levels.size() - 1;
    const std::size_t current_size = levels[current_level].size();
    auto& next = levels.emplace_back(current_size / 2);
    for (std::size_t i = 0; i < next.size(); ++i) {
      next[i] = hash_pair_cpu(levels[current_level][2 * i],
                              levels[current_level][2 * i + 1]);
    }
  }
  return levels;
}

}  // namespace

std::vector<Leaf> deterministic_leaves(std::size_t count, std::uint64_t seed) {
  std::vector<Leaf> leaves(count);
  std::uint64_t x = seed;
  for (auto& leaf : leaves) {
    for (auto& value : leaf) {
      x = splitmix64(x);
      value = x % field::kModulus;
    }
  }
  return leaves;
}

Digest hash_leaf_cpu(const Leaf& leaf) {
  State state{};
  for (std::size_t i = 0; i < leaf.size(); ++i) {
    state[i] = leaf[i];
  }
  state[kRateWords] = 1;
  poseidon2_permute(state);
  return digest_from_state(state);
}

Digest hash_pair_cpu(const Digest& left, const Digest& right) {
  State state{};
  for (std::size_t i = 0; i < left.size(); ++i) {
    state[i] = left[i];
    state[i + left.size()] = right[i];
  }
  state[kRateWords] = 2;
  poseidon2_permute(state);
  return digest_from_state(state);
}

ForestBuildResult merkle_forest_roots_cpu(std::span<const Leaf> leaves,
                                          ForestShape shape) {
  validate_shape(leaves, shape);
  const auto start = std::chrono::steady_clock::now();

  std::vector<Digest> current(leaves.size());
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    current[i] = hash_leaf_cpu(leaves[i]);
  }

  std::size_t level_count = shape.leaves_per_tree;
  while (level_count > 1) {
    const std::size_t parents_per_tree = level_count / 2;
    std::vector<Digest> next(shape.tree_count * parents_per_tree);
    for (std::size_t tree = 0; tree < shape.tree_count; ++tree) {
      for (std::size_t i = 0; i < parents_per_tree; ++i) {
        const auto& left = current[tree * level_count + 2 * i];
        const auto& right = current[tree * level_count + 2 * i + 1];
        next[tree * parents_per_tree + i] = hash_pair_cpu(left, right);
      }
    }
    current = std::move(next);
    level_count = parents_per_tree;
  }

  const auto end = std::chrono::steady_clock::now();
  ForestBuildResult result;
  result.roots = std::move(current);
  result.host_ms = std::chrono::duration<double, std::milli>(end - start).count();
  result.hashes = merkle_hash_count(shape);
  result.bytes_absorbed = absorbed_bytes_for_hashes(result.hashes);
  return result;
}

AuthPath merkle_auth_path_cpu(std::span<const Leaf> leaves,
                              ForestShape shape,
                              std::size_t tree_index,
                              std::size_t leaf_index) {
  validate_shape(leaves, shape);
  if (tree_index >= shape.tree_count || leaf_index >= shape.leaves_per_tree) {
    throw std::out_of_range("auth path index out of range");
  }

  const std::size_t offset = tree_index * shape.leaves_per_tree;
  const auto levels = build_tree_levels(leaves.subspan(offset, shape.leaves_per_tree));

  AuthPath path;
  path.tree_index = tree_index;
  path.leaf_index = leaf_index;
  path.leaf = leaves[offset + leaf_index];
  path.root = levels.back().front();

  std::size_t idx = leaf_index;
  for (std::size_t level = 0; level + 1 < levels.size(); ++level) {
    path.siblings.push_back(levels[level][idx ^ 1U]);
    idx >>= 1U;
  }
  return path;
}

std::vector<AuthPath> merkle_auth_paths_cpu(std::span<const Leaf> leaves,
                                            ForestShape shape,
                                            std::span<const std::size_t> tree_indices,
                                            std::span<const std::size_t> leaf_indices) {
  if (tree_indices.size() != leaf_indices.size()) {
    throw std::invalid_argument("auth path tree and leaf index counts differ");
  }
  std::vector<AuthPath> paths;
  paths.reserve(tree_indices.size());
  for (std::size_t i = 0; i < tree_indices.size(); ++i) {
    paths.push_back(merkle_auth_path_cpu(leaves, shape, tree_indices[i], leaf_indices[i]));
  }
  return paths;
}

bool verify_auth_path_cpu(const AuthPath& path) {
  Digest current = hash_leaf_cpu(path.leaf);
  std::size_t idx = path.leaf_index;
  for (const auto& sibling : path.siblings) {
    current = (idx & 1U) == 0 ? hash_pair_cpu(current, sibling)
                              : hash_pair_cpu(sibling, current);
    idx >>= 1U;
  }
  return current == path.root;
}

std::size_t merkle_hash_count(ForestShape shape) {
  return shape.tree_count * (2 * shape.leaves_per_tree - 1);
}

std::size_t absorbed_bytes_for_hashes(std::size_t hashes) {
  return hashes * kRateWords * sizeof(std::uint64_t);
}

}  // namespace cpb::poseidon2
