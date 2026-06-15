#include "cpb/vector_ops.hpp"

#include <stdexcept>

#include "cpb/field.hpp"

namespace cpb {

std::vector<std::uint64_t> vector_add_cpu(const std::vector<std::uint64_t>& a,
                                          const std::vector<std::uint64_t>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("vector_add_cpu input sizes differ");
  }
  std::vector<std::uint64_t> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = field::add(a[i], b[i]);
  }
  return out;
}

std::vector<std::uint64_t> vector_mul_cpu(const std::vector<std::uint64_t>& a,
                                          const std::vector<std::uint64_t>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("vector_mul_cpu input sizes differ");
  }
  std::vector<std::uint64_t> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = field::mul(a[i], b[i]);
  }
  return out;
}

}  // namespace cpb
