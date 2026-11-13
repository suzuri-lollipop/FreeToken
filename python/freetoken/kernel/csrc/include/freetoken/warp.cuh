#pragma once
#include <freetoken/utils.cuh>

// glibc-only, and nothing below needs it; MSVC has no sys/cdefs.h
#ifndef _WIN32
#include <sys/cdefs.h>
#endif

#include <cstddef>

namespace device::warp {

namespace details {

template <std::size_t kUnit> inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4,
                  "Unsupported memory package size");
  }
}

inline constexpr auto resolve_unit_size(std::size_t x) -> std::size_t {
  if (x % (16 * kWarpThreads) == 0)
    return 16;
  if (x % (8 * kWarpThreads) == 0)
    return 8;
  if (x % (4 * kWarpThreads) == 0)
    return 4;
  // sub-warp copy (x < 128 bytes): the widest package that still tiles x, so a
  // partial warp (x / unit lanes) moves it instead of asserting on a short row
  if (x % 16 == 0)
    return 16;
  if (x % 8 == 0)
    return 8;
  if (x % 4 == 0)
    return 4;
  return 0; // trigger static assert in _get_mem_package
}

template <std::size_t kBytes, std::size_t kUnit>
using mem_package_t = decltype(get_mem_package<kUnit>());

} // namespace details

template <std::size_t kBytes,
          std::size_t kUnit = details::resolve_unit_size(kBytes)>
__always_inline __device__ void copy(void *__restrict__ dst,
                                     const void *__restrict__ src) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kTotalPackages = kBytes / sizeof(Package);
  static_assert(kBytes % sizeof(Package) == 0,
                "kBytes must be a multiple of the memory package");

  const auto dst_packed = static_cast<Package *>(dst);
  const auto src_packed = static_cast<const Package *>(src);
  const auto lane_id = threadIdx.x % kWarpThreads;

  // a full warp strides over the packages; a sub-warp row (kTotalPackages <
  // kWarpThreads) simply leaves the high lanes idle
#pragma unroll
  for (std::size_t j = lane_id; j < kTotalPackages; j += kWarpThreads) {
    dst_packed[j] = src_packed[j];
  }
}

template <std::size_t kBytes,
          std::size_t kUnit = details::resolve_unit_size(kBytes)>
__always_inline __device__ void reset(void *__restrict__ dst) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kTotalPackages = kBytes / sizeof(Package);
  static_assert(kBytes % sizeof(Package) == 0,
                "warp_copy: kBytes must be a multiple of the memory package");

  const auto dst_ = static_cast<Package *>(dst);
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto zero_value = Package{};

#pragma unroll
  for (std::size_t j = lane_id; j < kTotalPackages; j += kWarpThreads) {
    dst_[j] = zero_value;
  }
}

} // namespace device::warp
