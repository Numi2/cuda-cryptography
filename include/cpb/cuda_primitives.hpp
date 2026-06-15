#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cpb/mlkem.hpp"
#include "cpb/poseidon2_merkle.hpp"
#include "cpb/sha256.hpp"

namespace cpb::cuda {

struct DeviceInfo {
  int device = 0;
  std::string name;
  int runtime_version = 0;
  int driver_version = 0;
  int major = 0;
  int minor = 0;
  std::size_t global_memory_bytes = 0;
};

struct CudaBenchmarkResult {
  double device_ms = 0.0;
  std::size_t operations = 0;
  std::size_t bytes = 0;
};

bool is_available();
DeviceInfo device_info();

std::vector<std::uint64_t> vector_add(const std::vector<std::uint64_t>& a,
                                      const std::vector<std::uint64_t>& b);

std::vector<std::uint64_t> vector_mul(const std::vector<std::uint64_t>& a,
                                      const std::vector<std::uint64_t>& b);

std::vector<std::uint64_t> ntt(const std::vector<std::uint64_t>& values,
                               bool inverse = false);

Sha256Digest merkle_root(const std::vector<std::uint64_t>& leaves);

mlkem::Poly mlkem_poly_mul_ntt(const mlkem::Poly& a, const mlkem::Poly& b);

std::vector<mlkem::Poly> mlkem_ntt_batch(const std::vector<mlkem::Poly>& polys,
                                         bool inverse = false);

std::vector<mlkem::Poly> mlkem_poly_mul_ntt_batch(
    const std::vector<mlkem::Poly>& a,
    const std::vector<mlkem::Poly>& b);

CudaBenchmarkResult mlkem_ntt_batch_resident_benchmark(
    const std::vector<mlkem::Poly>& polys,
    int iterations,
    bool inverse = false);

CudaBenchmarkResult mlkem_poly_mul_ntt_batch_resident_benchmark(
    const std::vector<mlkem::Poly>& a,
    const std::vector<mlkem::Poly>& b,
    int iterations);

poseidon2::ForestBuildResult poseidon2_merkle_forest(
    const std::vector<poseidon2::Leaf>& leaves,
    poseidon2::ForestShape shape);

poseidon2::ForestBuildResult poseidon2_merkle_forest_resident_benchmark(
    const std::vector<poseidon2::Leaf>& leaves,
    poseidon2::ForestShape shape,
    int iterations);

}  // namespace cpb::cuda
