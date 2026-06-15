#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "cpb/field.hpp"
#include "cpb/mlkem.hpp"
#include "cpb/merkle.hpp"
#include "cpb/ntt.hpp"
#include "cpb/poseidon2_merkle.hpp"
#include "cpb/sha256.hpp"
#include "cpb/vector_ops.hpp"

#ifdef CPB_WITH_CUDA
#include "cpb/cuda_primitives.hpp"
#endif

namespace {

std::vector<std::uint64_t> deterministic_values(std::size_t n, std::uint64_t seed) {
  std::vector<std::uint64_t> values(n);
  std::uint64_t x = seed;
  for (std::size_t i = 0; i < n; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    values[i] = x % cpb::field::kModulus;
  }
  return values;
}

std::string hex_digest(const cpb::Sha256Digest& digest) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const auto byte : digest) {
    out << std::setw(2) << static_cast<int>(byte);
  }
  return out.str();
}

bool expect_equal(const std::vector<std::uint64_t>& expected,
                  const std::vector<std::uint64_t>& actual,
                  const std::string& name) {
  if (expected.size() != actual.size()) {
    std::cerr << name << " size mismatch: expected " << expected.size() << ", got "
              << actual.size() << '\n';
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      std::cerr << name << " first mismatch at index " << i << ": expected "
                << expected[i] << ", got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

bool expect_equal(const cpb::Sha256Digest& expected,
                  const cpb::Sha256Digest& actual,
                  const std::string& name) {
  if (expected != actual) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != actual[i]) {
        std::cerr << name << " first mismatch at byte " << i << ": expected 0x"
                  << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(expected[i]) << ", got 0x" << std::setw(2)
                  << static_cast<int>(actual[i]) << std::dec << '\n';
        return false;
      }
    }
  }
  return true;
}

bool expect_equal(const cpb::mlkem::Poly& expected,
                  const cpb::mlkem::Poly& actual,
                  const std::string& name) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      std::cerr << name << " first mismatch at coefficient " << i << ": expected "
                << expected[i] << ", got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

bool expect_equal(const std::vector<cpb::mlkem::Poly>& expected,
                  const std::vector<cpb::mlkem::Poly>& actual,
                  const std::string& name) {
  if (expected.size() != actual.size()) {
    std::cerr << name << " batch size mismatch: expected " << expected.size()
              << ", got " << actual.size() << '\n';
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!expect_equal(expected[i], actual[i], name + " poly=" + std::to_string(i))) {
      return false;
    }
  }
  return true;
}

bool expect_equal(const cpb::poseidon2::Digest& expected,
                  const cpb::poseidon2::Digest& actual,
                  const std::string& name) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      std::cerr << name << " first mismatch at word " << i << ": expected "
                << expected[i] << ", got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

bool expect_equal(const std::vector<cpb::poseidon2::Digest>& expected,
                  const std::vector<cpb::poseidon2::Digest>& actual,
                  const std::string& name) {
  if (expected.size() != actual.size()) {
    std::cerr << name << " root count mismatch: expected " << expected.size()
              << ", got " << actual.size() << '\n';
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!expect_equal(expected[i], actual[i], name + " root=" + std::to_string(i))) {
      return false;
    }
  }
  return true;
}

template <typename Fn>
bool expect_throws(Fn&& fn, const std::string& name) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  std::cerr << name << " expected an exception\n";
  return false;
}

bool test_field_vectors() {
  const std::uint64_t p = cpb::field::kModulus;
  bool ok = true;
  ok &= cpb::field::add(p - 1, 2) == 1;
  ok &= cpb::field::sub(1, 2) == p - 1;
  ok &= cpb::field::mul(p - 1, p - 1) == 1;
  ok &= cpb::field::pow(cpb::field::kPrimitiveRoot, p - 1) == 1;
  ok &= cpb::field::root_of_unity(1024) != 1;
  ok &= cpb::field::pow(cpb::field::root_of_unity(1024), 1024) == 1;
  if (!ok) {
    std::cerr << "field deterministic vector test failed\n";
  }
  return ok;
}

bool test_sha256_vectors() {
  const std::array<std::uint8_t, 3> abc = {'a', 'b', 'c'};
  const auto digest = cpb::sha256(abc);
  const std::string expected =
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  if (hex_digest(digest) != expected) {
    std::cerr << "sha256('abc') mismatch: expected " << expected << ", got "
              << hex_digest(digest) << '\n';
    return false;
  }
  return true;
}

bool test_cpu_primitives() {
  const auto a = deterministic_values(1024, 1);
  const auto b = deterministic_values(1024, 2);
  const auto add = cpb::vector_add_cpu(a, b);
  const auto mul = cpb::vector_mul_cpu(a, b);

  bool ok = true;
  ok &= add[0] == cpb::field::add(a[0], b[0]);
  ok &= mul[17] == cpb::field::mul(a[17], b[17]);

  auto transformed = cpb::ntt_cpu(a);
  auto roundtrip = cpb::ntt_cpu(transformed, true);
  ok &= expect_equal(a, roundtrip, "cpu ntt roundtrip");

  const auto root = cpb::merkle_root_cpu(std::vector<std::uint64_t>{0, 1, 2, 3, 4});
  const std::string expected_root =
      "0eb71b848fb38e0cba18edcd015ce98116dfb4fd620f1063e7ecf65c80e13db7";
  if (hex_digest(root) != expected_root) {
    std::cerr << "merkle deterministic vector mismatch: expected " << expected_root
              << ", got " << hex_digest(root) << '\n';
    ok = false;
  }

  if (!ok) {
    std::cerr << "CPU primitive test failed\n";
  }
  return ok;
}

bool test_mlkem_cpu_primitives() {
  bool ok = true;
  ok &= cpb::mlkem::kModulus == 3329;
  ok &= cpb::mlkem::pow_mod(cpb::mlkem::kPrimitiveRoot256, 256) == 1;
  ok &= cpb::mlkem::pow_mod(cpb::mlkem::kPrimitiveRoot256, 128) ==
        cpb::mlkem::kModulus - 1;

  const auto a = cpb::mlkem::deterministic_poly(123);
  const auto b = cpb::mlkem::deterministic_poly(456);
  const std::array<std::uint16_t, 8> expected_a = {
      426, 1911, 1974, 179, 329, 2496, 2804, 328};
  const std::array<std::uint16_t, 8> expected_b = {
      2093, 617, 1521, 25, 919, 2890, 1385, 356};
  for (std::size_t i = 0; i < expected_a.size(); ++i) {
    ok &= a[i] == expected_a[i];
    ok &= b[i] == expected_b[i];
  }

  const auto roundtrip = cpb::mlkem::ntt(cpb::mlkem::ntt(a), true);
  ok &= expect_equal(a, roundtrip, "mlkem cpu ntt roundtrip");

  const auto schoolbook = cpb::mlkem::poly_mul_schoolbook(a, b);
  const auto ntt = cpb::mlkem::poly_mul_ntt(a, b);
  ok &= expect_equal(schoolbook, ntt, "mlkem cpu ntt mul");

  const std::array<std::uint16_t, 16> expected_product = {
      2684, 2203, 3003, 2843, 2176, 2360, 2144, 943,
      1820, 2594, 2416, 2662, 369, 3155, 2020, 2920};
  for (std::size_t i = 0; i < expected_product.size(); ++i) {
    if (schoolbook[i] != expected_product[i]) {
      std::cerr << "mlkem deterministic product mismatch at coefficient " << i
                << ": expected " << expected_product[i] << ", got " << schoolbook[i]
                << '\n';
      ok = false;
      break;
    }
  }

  if (!ok) {
    std::cerr << "ML-KEM-style CPU primitive test failed\n";
  }
  return ok;
}

bool test_poseidon2_merkle_cpu() {
  const cpb::poseidon2::ForestShape shape{2, 8};
  const auto leaves = cpb::poseidon2::deterministic_leaves(
      shape.tree_count * shape.leaves_per_tree, 999);
  const auto result = cpb::poseidon2::merkle_forest_roots_cpu(leaves, shape);

  bool ok = true;
  const auto vector_leaves = cpb::poseidon2::deterministic_leaves(4, 777);
  const auto leaf0 = cpb::poseidon2::hash_leaf_cpu(vector_leaves[0]);
  const auto leaf1 = cpb::poseidon2::hash_leaf_cpu(vector_leaves[1]);
  const auto pair01 = cpb::poseidon2::hash_pair_cpu(leaf0, leaf1);
  const auto small_forest =
      cpb::poseidon2::merkle_forest_roots_cpu(vector_leaves, {1, 4});
  ok &= expect_equal({9663374225833733844ULL, 14846115997077300002ULL,
                      3874052560956134737ULL, 18351804075086715189ULL},
                     leaf0, "poseidon2 leaf hash vector 0");
  ok &= expect_equal({15967017389795311074ULL, 12745275773536680036ULL,
                      18159984983286535455ULL, 703431422288797923ULL},
                     leaf1, "poseidon2 leaf hash vector 1");
  ok &= expect_equal({167455533063635113ULL, 11393647405094625292ULL,
                      9588948309221566780ULL, 6402483533664955761ULL},
                     pair01, "poseidon2 pair hash vector");
  ok &= expect_equal({5921759479114827537ULL, 16802967967082501393ULL,
                      766461081830350271ULL, 8482344946004106393ULL},
                     small_forest.roots.front(), "poseidon2 small forest root vector");

  ok &= result.roots.size() == shape.tree_count;
  ok &= result.hashes == cpb::poseidon2::merkle_hash_count(shape);
  ok &= result.bytes_absorbed == cpb::poseidon2::absorbed_bytes_for_hashes(result.hashes);
  const std::vector<cpb::poseidon2::Digest> expected_roots = {
      {2833046405353280632ULL, 17455829047229495281ULL, 3448498022770064707ULL,
       1737143749959670836ULL},
      {7507967109803505144ULL, 8818948803314405175ULL, 18362079593053762419ULL,
       837974301746350213ULL},
  };
  ok &= expect_equal(expected_roots, result.roots, "poseidon2 deterministic roots");

  const auto path = cpb::poseidon2::merkle_auth_path_cpu(leaves, shape, 1, 5);
  ok &= cpb::poseidon2::verify_auth_path_cpu(path);

  const std::array<std::size_t, 4> trees = {0, 0, 1, 1};
  const std::array<std::size_t, 4> indices = {0, 7, 2, 6};
  const auto paths = cpb::poseidon2::merkle_auth_paths_cpu(leaves, shape, trees, indices);
  ok &= paths.size() == trees.size();
  for (const auto& auth_path : paths) {
    ok &= cpb::poseidon2::verify_auth_path_cpu(auth_path);
  }

  ok &= expect_throws(
      [&] { cpb::poseidon2::merkle_forest_roots_cpu(leaves, {0, 8}); },
      "poseidon2 rejects zero tree count");
  ok &= expect_throws(
      [&] { cpb::poseidon2::merkle_forest_roots_cpu(leaves, {1, 0}); },
      "poseidon2 rejects zero leaves per tree");
  ok &= expect_throws(
      [&] { cpb::poseidon2::merkle_forest_roots_cpu(leaves, {1, 3}); },
      "poseidon2 rejects non-power-of-two leaves");
  ok &= expect_throws(
      [&] {
        cpb::poseidon2::merkle_forest_roots_cpu(
            std::span<const cpb::poseidon2::Leaf>(leaves.data(), leaves.size() - 1),
            shape);
      },
      "poseidon2 rejects mismatched leaf count");
  ok &= expect_throws(
      [&] {
        const std::array<std::size_t, 2> bad_trees = {0, 1};
        const std::array<std::size_t, 1> bad_indices = {0};
        cpb::poseidon2::merkle_auth_paths_cpu(leaves, shape, bad_trees, bad_indices);
      },
      "poseidon2 rejects mismatched auth-path index counts");

  if (!ok) {
    std::cerr << "Poseidon2-style Merkle forest CPU test failed\n";
  }
  return ok;
}

bool test_cuda_primitives() {
#ifdef CPB_WITH_CUDA
  if (!cpb::cuda::is_available()) {
    std::cout << "CUDA runtime built, but no CUDA device is visible; skipping CUDA comparisons\n";
    return true;
  }

  bool ok = true;
  ok &= expect_equal(std::vector<std::uint64_t>{}, cpb::cuda::vector_add({}, {}),
                     "cuda empty vector_add");
  ok &= expect_equal(std::vector<std::uint64_t>{}, cpb::cuda::vector_mul({}, {}),
                     "cuda empty vector_mul");
  ok &= expect_equal(std::vector<cpb::mlkem::Poly>{}, cpb::cuda::mlkem_ntt_batch({}),
                     "cuda empty mlkem ntt batch");
  ok &= expect_equal(std::vector<cpb::mlkem::Poly>{},
                     cpb::cuda::mlkem_poly_mul_ntt_batch({}, {}),
                     "cuda empty mlkem poly mul batch");
  ok &= expect_throws(
      [&] { cpb::cuda::vector_add({1}, {}); },
      "cuda vector_add rejects mismatched sizes");
  ok &= expect_throws(
      [&] { cpb::cuda::vector_mul({1}, {}); },
      "cuda vector_mul rejects mismatched sizes");
  ok &= expect_throws(
      [&] { cpb::cuda::poseidon2_merkle_forest({}, {0, 0}); },
      "cuda poseidon2 rejects empty shape");

  for (const std::size_t n : {1UL, 2UL, 1024UL, 4096UL}) {
    const auto a = deterministic_values(n, 11);
    const auto b = deterministic_values(n, 12);
    ok &= expect_equal(cpb::vector_add_cpu(a, b), cpb::cuda::vector_add(a, b),
                       "cuda vector_add n=" + std::to_string(n));
    ok &= expect_equal(cpb::vector_mul_cpu(a, b), cpb::cuda::vector_mul(a, b),
                       "cuda vector_mul n=" + std::to_string(n));
  }

  for (const std::size_t n : {2UL, 16UL, 1024UL}) {
    const auto values = deterministic_values(n, 21);
    ok &= expect_equal(cpb::ntt_cpu(values), cpb::cuda::ntt(values),
                       "cuda ntt n=" + std::to_string(n));
    ok &= expect_equal(values, cpb::cuda::ntt(cpb::cuda::ntt(values), true),
                       "cuda ntt roundtrip n=" + std::to_string(n));
  }

  const auto leaves = deterministic_values(2048, 31);
  ok &= expect_equal(cpb::merkle_root_cpu(leaves), cpb::cuda::merkle_root(leaves),
                     "cuda merkle root");

  const auto mlkem_a = cpb::mlkem::deterministic_poly(123);
  const auto mlkem_b = cpb::mlkem::deterministic_poly(456);
  ok &= expect_equal(cpb::mlkem::poly_mul_ntt(mlkem_a, mlkem_b),
                     cpb::cuda::mlkem_poly_mul_ntt(mlkem_a, mlkem_b),
                     "cuda mlkem single poly mul");

  std::vector<cpb::mlkem::Poly> batch_a(32);
  std::vector<cpb::mlkem::Poly> batch_b(32);
  std::vector<cpb::mlkem::Poly> expected_ntt(32);
  std::vector<cpb::mlkem::Poly> expected_mul(32);
  for (std::size_t i = 0; i < batch_a.size(); ++i) {
    batch_a[i] = cpb::mlkem::deterministic_poly(static_cast<std::uint32_t>(1000 + i));
    batch_b[i] = cpb::mlkem::deterministic_poly(static_cast<std::uint32_t>(2000 + i));
    expected_ntt[i] = cpb::mlkem::ntt(batch_a[i]);
    expected_mul[i] = cpb::mlkem::poly_mul_ntt(batch_a[i], batch_b[i]);
  }
  ok &= expect_equal(expected_ntt, cpb::cuda::mlkem_ntt_batch(batch_a),
                     "cuda mlkem ntt batch");
  ok &= expect_equal(expected_mul, cpb::cuda::mlkem_poly_mul_ntt_batch(batch_a, batch_b),
                     "cuda mlkem poly mul batch");
  ok &= expect_throws(
      [&] {
        std::vector<cpb::mlkem::Poly> shorter(31);
        cpb::cuda::mlkem_poly_mul_ntt_batch(batch_a, shorter);
      },
      "cuda mlkem rejects mismatched batch sizes");

  const cpb::poseidon2::ForestShape p2_shape{4, 16};
  const auto p2_leaves = cpb::poseidon2::deterministic_leaves(
      p2_shape.tree_count * p2_shape.leaves_per_tree, 12345);
  const auto p2_cpu = cpb::poseidon2::merkle_forest_roots_cpu(p2_leaves, p2_shape);
  const auto p2_cuda = cpb::cuda::poseidon2_merkle_forest(p2_leaves, p2_shape);
  ok &= expect_equal(p2_cpu.roots, p2_cuda.roots, "cuda poseidon2 merkle forest");
  return ok;
#else
  std::cout << "Built without CUDA; CPU correctness checks completed\n";
  return true;
#endif
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok &= test_field_vectors();
    ok &= test_sha256_vectors();
    ok &= test_cpu_primitives();
    ok &= test_mlkem_cpu_primitives();
    ok &= test_poseidon2_merkle_cpu();
    ok &= test_cuda_primitives();

    if (!ok) {
      return 1;
    }
    std::cout << "All correctness tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << '\n';
    return 1;
  }
}
