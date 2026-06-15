#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cpb/field.hpp"
#include "cpb/merkle.hpp"
#include "cpb/ntt.hpp"
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

bool test_cuda_primitives() {
#ifdef CPB_WITH_CUDA
  if (!cpb::cuda::is_available()) {
    std::cout << "CUDA runtime built, but no CUDA device is visible; skipping CUDA comparisons\n";
    return true;
  }

  bool ok = true;
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
