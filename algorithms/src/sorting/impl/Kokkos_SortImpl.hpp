//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_SORT_FREE_FUNCS_IMPL_HPP_
#define KOKKOS_SORT_FREE_FUNCS_IMPL_HPP_

#include "../Kokkos_BinOpsPublicAPI.hpp"
#include "../Kokkos_BinSortPublicAPI.hpp"
#include <std_algorithms/Kokkos_BeginEnd.hpp>
#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)

// Workaround for `Instruction 'shfl' without '.sync' is not supported on
// .target sm_70 and higher from PTX ISA version 6.4`.
// Also see https://github.com/NVIDIA/cub/pull/170.
#if !defined(CUB_USE_COOPERATIVE_GROUPS)
#define CUB_USE_COOPERATIVE_GROUPS
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

#if defined(KOKKOS_COMPILER_CLANG)
// Some versions of Clang fail to compile Thrust, failing with errors like
// this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
// The exact combination of versions for Clang and Thrust (or CUDA) for this
// failure was not investigated, however even very recent version combination
// (Clang 10.0.0 and Cuda 10.0) demonstrated failure.
//
// Defining _CubLog here locally allows us to avoid that code path, however
// disabling some debugging diagnostics
#pragma push_macro("_CubLog")
#ifdef _CubLog
#undef _CubLog
#endif
#define _CubLog
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#pragma pop_macro("_CubLog")
#else
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

#pragma GCC diagnostic pop

#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#endif

namespace Kokkos {
namespace Impl {

template <class ViewType>
struct min_max_functor {
  using minmax_scalar =
      Kokkos::MinMaxScalar<typename ViewType::non_const_value_type>;

  ViewType view;
  min_max_functor(const ViewType& view_) : view(view_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i, minmax_scalar& minmax) const {
    if (view(i) < minmax.min_val) minmax.min_val = view(i);
    if (view(i) > minmax.max_val) minmax.max_val = view(i);
  }
};

template <class ExecutionSpace, class DataType, class... Properties>
void sort_via_binsort(const ExecutionSpace& exec,
                      const Kokkos::View<DataType, Properties...>& view) {
  // Although we are using BinSort below, which could work on rank-2 views,
  // for now view must be rank-1 because the min_max_functor
  // used below only works for rank-1 views
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(ViewType::rank == 1,
                "Kokkos::sort: currently only supports rank-1 Views.");

  if (view.extent(0) == 0) {
    return;
  }

  Kokkos::MinMaxScalar<typename ViewType::non_const_value_type> result;
  Kokkos::MinMax<typename ViewType::non_const_value_type> reducer(result);
  parallel_reduce("Kokkos::Sort::FindExtent",
                  Kokkos::RangePolicy<typename ViewType::execution_space>(
                      exec, 0, view.extent(0)),
                  min_max_functor<ViewType>(view), reducer);
  if (result.min_val == result.max_val) return;
  // For integral types the number of bins may be larger than the range
  // in which case we can exactly have one unique value per bin
  // and then don't need to sort bins.
  bool sort_in_bins = true;
  // TODO: figure out better max_bins then this ...
  int64_t max_bins = view.extent(0) / 2;
  if (std::is_integral<typename ViewType::non_const_value_type>::value) {
    // Cast to double to avoid possible overflow when using integer
    auto const max_val = static_cast<double>(result.max_val);
    auto const min_val = static_cast<double>(result.min_val);
    // using 10M as the cutoff for special behavior (roughly 40MB for the count
    // array)
    if ((max_val - min_val) < 10000000) {
      max_bins     = max_val - min_val + 1;
      sort_in_bins = false;
    }
  }
  if (std::is_floating_point<typename ViewType::non_const_value_type>::value) {
    KOKKOS_ASSERT(std::isfinite(static_cast<double>(result.max_val) -
                                static_cast<double>(result.min_val)));
  }

  using CompType = BinOp1D<ViewType>;
  BinSort<ViewType, CompType> bin_sort(
      view, CompType(max_bins, result.min_val, result.max_val), sort_in_bins);
  bin_sort.create_permute_vector(exec);
  bin_sort.sort(exec, view);
}

#if defined(KOKKOS_ENABLE_CUDA)
template <class DataType, class... Properties>
void sort_cudathrust(const Cuda& space,
                     const Kokkos::View<DataType, Properties...>& view) {
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(ViewType::rank == 1,
                "Kokkos::sort: currently only supports rank-1 Views.");

  if (view.extent(0) == 0) {
    return;
  }
  const auto exec = thrust::cuda::par.on(space.cuda_stream());
  auto first      = Experimental::begin(view);
  auto last       = Experimental::end(view);
  thrust::sort(exec, first, last);
}
#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
template <class DataType, class... Properties>
void sort_odedpl(const Experimental::SYCL& space,
                 const Kokkos::View<DataType, Properties...>& view) {
  using ViewType = Kokkos::View<DataType, Properties...>;
  static_assert(SpaceAccessibility<Experimental::SYCL,
                                   typename ViewType::memory_space>::accessible,
                "SYCL execution space is not able to access the memory space "
                "of the View argument!");

  // Can't use Experimental::begin/end here since the oneDPL then assumes that
  // the data is on the host.
  static_assert(
      ViewType::rank == 1 &&
          (std::is_same<typename ViewType::array_layout, LayoutRight>::value ||
           std::is_same<typename ViewType::array_layout, LayoutLeft>::value),
      "SYCL sort only supports contiguous rank-1 Views.");

  if (view.extent(0) == 0) {
    return;
  }

  auto queue  = space.sycl_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(queue);
  const int n = view.extent(0);
  oneapi::dpl::sort(policy, view.data(), view.data() + n);
}
#endif

// --------------------------------------------------
//
// specialize cases for sorting without comparator
//
// --------------------------------------------------

#if defined(KOKKOS_ENABLE_CUDA)
template <class DataType, class... Properties>
void sort_device_view_without_comparator(
    const Cuda& exec, const Kokkos::View<DataType, Properties...>& view) {
  sort_cudathrust(exec, view);
}
#endif

#if defined(KOKKOS_ENABLE_ONEDPL)
template <class DataType, class... Properties>
void sort_device_view_without_comparator(
    const Experimental::SYCL& exec,
    const Kokkos::View<DataType, Properties...>& view) {
  sort_onedpl(exec, view);
}
#endif

// fallback case
template <class ExecutionSpace, class DataType, class... Properties>
std::enable_if_t<Kokkos::is_execution_space<ExecutionSpace>::value>
sort_device_view_without_comparator(
    const ExecutionSpace& exec,
    const Kokkos::View<DataType, Properties...>& view) {
  sort_via_binsort(exec, view);
}

}  // namespace Impl
}  // namespace Kokkos
#endif
