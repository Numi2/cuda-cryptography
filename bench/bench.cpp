#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cpb/field.hpp"
#include "cpb/mlkem.hpp"
#include "cpb/merkle.hpp"
#include "cpb/ntt.hpp"
#include "cpb/vector_ops.hpp"

#ifdef CPB_WITH_CUDA
#include "cpb/cuda_primitives.hpp"
#endif

namespace {

using Clock = std::chrono::steady_clock;

volatile std::uint64_t g_sink = 0;

std::vector<std::uint64_t> deterministic_values(std::size_t n, std::uint64_t seed) {
  std::vector<std::uint64_t> values(n);
  std::uint64_t x = seed;
  for (std::size_t i = 0; i < n; ++i) {
    x = x * 2862933555777941757ULL + 3037000493ULL;
    values[i] = x % cpb::field::kModulus;
  }
  return values;
}

template <typename Fn>
double time_ms(Fn&& fn, int iterations = 3) {
  fn();
  const auto start = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    fn();
  }
  const auto end = Clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

void print_row(const std::string& backend,
               const std::string& primitive,
               std::size_t n,
               double ms,
               double throughput) {
  std::cout << "| " << std::left << std::setw(6) << backend << " | " << std::setw(13)
            << primitive << " | " << std::right << std::setw(8) << n << " | "
            << std::setw(10) << std::fixed << std::setprecision(3) << ms << " | "
            << std::setw(14) << std::setprecision(2) << throughput << " |\n";
}

void print_mlkem_row(const std::string& backend,
                     const std::string& primitive,
                     const std::string& workload,
                     double ms,
                     double throughput) {
  std::cout << "| " << std::left << std::setw(6) << backend << " | " << std::setw(19)
            << primitive << " | " << std::setw(12) << workload << " | "
            << std::right << std::setw(10) << std::fixed << std::setprecision(3) << ms
            << " | " << std::setw(18) << std::setprecision(3) << throughput << " |\n";
}

std::vector<cpb::mlkem::Poly> deterministic_poly_batch(std::size_t batch,
                                                       std::uint32_t seed) {
  std::vector<cpb::mlkem::Poly> out(batch);
  for (std::size_t i = 0; i < batch; ++i) {
    out[i] = cpb::mlkem::deterministic_poly(seed + static_cast<std::uint32_t>(i));
  }
  return out;
}

void run_cpu_benchmarks() {
  for (const std::size_t n : {1024UL, 4096UL, 16384UL, 65536UL}) {
    const auto a = deterministic_values(n, 1);
    const auto b = deterministic_values(n, 2);
    const double add_ms = time_ms([&] {
      const auto out = cpb::vector_add_cpu(a, b);
      g_sink ^= out.front();
    }, 5);
    print_row("CPU", "field add", n, add_ms, (static_cast<double>(n) / add_ms) / 1000.0);
    const double mul_ms = time_ms([&] {
      const auto out = cpb::vector_mul_cpu(a, b);
      g_sink ^= out.front();
    }, 5);
    print_row("CPU", "field mul", n, mul_ms, (static_cast<double>(n) / mul_ms) / 1000.0);
  }

  for (const std::size_t n : {1024UL, 4096UL, 16384UL}) {
    const auto values = deterministic_values(n, 3);
    const double ms = time_ms([&] {
      const auto out = cpb::ntt_cpu(values);
      g_sink ^= out.front();
    }, 3);
    print_row("CPU", "radix-2 ntt", n, ms, (static_cast<double>(n) / ms) / 1000.0);
  }

  for (const std::size_t n : {1024UL, 4096UL, 16384UL}) {
    const auto leaves = deterministic_values(n, 4);
    const double ms = time_ms([&] {
      const auto out = cpb::merkle_root_cpu(leaves);
      g_sink ^= out.front();
    }, 3);
    print_row("CPU", "merkle sha256", n, ms, (static_cast<double>(n) / ms) / 1000.0);
  }
}

void run_mlkem_cpu_benchmarks() {
  const auto poly = cpb::mlkem::deterministic_poly(501);
  const double ms = time_ms([&] {
    const auto out = cpb::mlkem::ntt(poly);
    g_sink ^= out.front();
  }, 1000);
  print_mlkem_row("CPU", "ML-KEM NTT", "n=256", ms, 1.0 / ms);
}

void run_cuda_benchmarks() {
#ifdef CPB_WITH_CUDA
  if (!cpb::cuda::is_available()) {
    std::cout << "\nCUDA backend was built, but no CUDA device is visible. Skipping GPU rows.\n";
    return;
  }

  for (const std::size_t n : {1024UL, 4096UL, 16384UL, 65536UL}) {
    const auto a = deterministic_values(n, 11);
    const auto b = deterministic_values(n, 12);
    const double add_ms = time_ms([&] {
      const auto out = cpb::cuda::vector_add(a, b);
      g_sink ^= out.front();
    }, 5);
    print_row("CUDA", "field add", n, add_ms, (static_cast<double>(n) / add_ms) / 1000.0);
    const double mul_ms = time_ms([&] {
      const auto out = cpb::cuda::vector_mul(a, b);
      g_sink ^= out.front();
    }, 5);
    print_row("CUDA", "field mul", n, mul_ms, (static_cast<double>(n) / mul_ms) / 1000.0);
  }

  for (const std::size_t n : {1024UL, 4096UL, 16384UL}) {
    const auto values = deterministic_values(n, 13);
    const double ms = time_ms([&] {
      const auto out = cpb::cuda::ntt(values);
      g_sink ^= out.front();
    }, 3);
    print_row("CUDA", "radix-2 ntt", n, ms, (static_cast<double>(n) / ms) / 1000.0);
  }

  for (const std::size_t n : {1024UL, 4096UL, 16384UL}) {
    const auto leaves = deterministic_values(n, 14);
    const double ms = time_ms([&] {
      const auto out = cpb::cuda::merkle_root(leaves);
      g_sink ^= out.front();
    }, 3);
    print_row("CUDA", "merkle sha256", n, ms, (static_cast<double>(n) / ms) / 1000.0);
  }
#else
  std::cout << "\nBuilt without CUDA. Reconfigure on a CUDA machine for GPU benchmark rows.\n";
#endif
}

void run_mlkem_cuda_benchmarks() {
#ifdef CPB_WITH_CUDA
  if (!cpb::cuda::is_available()) {
    std::cout << "\nCUDA backend was built, but no CUDA device is visible. "
                 "Skipping ML-KEM GPU rows.\n";
    return;
  }

  for (const std::size_t batch_size : {1UL, 1024UL, 10000UL, 100000UL}) {
    const auto batch = deterministic_poly_batch(batch_size, 701);
    const int iterations = batch_size <= 1024 ? 10 : 3;
    const double batch_ntt_ms = time_ms([&] {
      const auto out = cpb::cuda::mlkem_ntt_batch(batch);
      g_sink ^= out.front().front();
    }, iterations);
    print_mlkem_row("CUDA", "ML-KEM NTT",
                    "batch=" + std::to_string(batch_size), batch_ntt_ms,
                    static_cast<double>(batch_size) / batch_ntt_ms);
  }

  for (const std::size_t batch_size : {1UL, 1024UL, 10000UL, 100000UL}) {
    const auto batch_a = deterministic_poly_batch(batch_size, 1701);
    const auto batch_b = deterministic_poly_batch(batch_size, 2701);
    const int iterations = batch_size <= 1024 ? 5 : 3;
    const double batch_mul_ms = time_ms([&] {
      const auto out = cpb::cuda::mlkem_poly_mul_ntt_batch(batch_a, batch_b);
      g_sink ^= out.front().front();
    }, iterations);
    print_mlkem_row("CUDA", "ML-KEM poly mul",
                    "batch=" + std::to_string(batch_size), batch_mul_ms,
                    static_cast<double>(batch_size) / batch_mul_ms);
  }
#else
  std::cout << "\nBuilt without CUDA. Reconfigure on a CUDA machine for ML-KEM GPU rows.\n";
#endif
}

}  // namespace

int main() {
  try {
    std::cout << "cuda-cryptography benchmark\n";
    std::cout << "Field modulus: 2^64 - 2^32 + 1 (Goldilocks)\n\n";
    std::cout << "| Backend | Primitive     |        n |  Time (ms) | Throughput Mops/s |\n";
    std::cout << "|--------:|---------------|---------:|-----------:|-----------------:|\n";
    run_cpu_benchmarks();
    run_cuda_benchmarks();

    std::cout << "\nML-KEM/Kyber-style batched benchmark\n";
    std::cout << "Modulus: q = 3329, degree n = 256, primitive 256th root = 17\n\n";
    std::cout << "| Backend | Primitive           | Workload     |  Time (ms) | Throughput Kpoly/s |\n";
    std::cout << "|--------:|---------------------|--------------|-----------:|-------------------:|\n";
    run_mlkem_cpu_benchmarks();
    run_mlkem_cuda_benchmarks();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Benchmark failed: " << e.what() << '\n';
    return 1;
  }
}
